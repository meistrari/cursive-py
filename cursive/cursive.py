import json
import time
from typing import Any, Callable, Generic, Optional, TypeVar
import os

import openai as openai_client
from anthropic import APIError

from cursive.build_input import get_function_call_directives
from cursive.custom_function_call import parse_custom_function_call
from cursive.stream import StreamTransformer
from cursive.usage.cohere import get_cohere_usage
from cursive.vendor.cohere import CohereClient, process_cohere_stream

from cursive.custom_types import (
    BaseModel,
    CompletionMessage,
    CompletionPayload,
    CreateChatCompletionResponseExtended,
    CursiveAskCost,
    CursiveAskModelResponse,
    CursiveAskOnToken,
    CursiveAskUsage,
    CursiveEnrichedAnswer,
    CursiveError,
    CursiveErrorCode,
    CursiveFunction,
    CursiveHook,
    CursiveHookPayload,
    CursiveSetupOptions,
    CursiveSetupOptionsExpand,
    CursiveModel,
)
from cursive.hookable import create_debugger, create_hooks
from cursive.pricing import resolve_pricing
from cursive.usage.anthropic import get_anthropic_usage
from cursive.usage.openai import get_openai_usage
from cursive.utils import filter_null_values, random_id, resguard
from cursive.vendor.anthropic import (
    AnthropicClient,
    process_anthropic_stream,
)
from cursive.vendor.index import resolve_vendor_from_model
from cursive.vendor.openai import process_openai_stream
from cursive.vendor.replicate import ReplicateClient

# TODO: Improve implementation architecture, this was a quick and dirty
class Cursive:
    options: CursiveSetupOptions

    def __init__(
        self,
        max_retries: Optional[int] = None,
        expand: Optional[dict[str, Any]] = None,
        debug: Optional[bool] = None,
        openai: Optional[dict[str, Any]] = None,
        anthropic: Optional[dict[str, Any]] = None,
        cohere: Optional[dict[str, Any]] = None,
        replicate: Optional[dict[str, Any]] = None,
    ):
        openai_client.api_key = (openai or {}).get('api_key') or \
            os.environ.get('OPENAI_API_KEY')
        anthropic_client = AnthropicClient((anthropic or {}).get('api_key') or \
            os.environ.get('ANTHROPIC_API_KEY'))
        cohere_client = CohereClient((cohere or {}).get('api_key') or \
            os.environ.get('CO_API_KEY', '---'))
        replicate_client = ReplicateClient((replicate or {}).get('api_key') or \
            os.environ.get('REPLICATE_API_TOKEN', '---'))

        self._hooks = create_hooks()
        self._vendor = CursiveVendors(
            openai=openai_client,
            anthropic=anthropic_client,
            cohere=cohere_client,
            replicate=replicate_client,
        )
        self.options = CursiveSetupOptions(
            max_retries=max_retries,
            expand=expand,
            debug=debug,
        )

        if debug:
            self._debugger = create_debugger(self._hooks, { 'tag': 'cursive' })


    def on(self, event: CursiveHook, callback: Callable):
        self._hooks.hook(event, callback)
        
    
    def ask(
        self,
        model: Optional[str | CursiveModel] = None,
        system_message: Optional[str] = None,
        functions: Optional[list[CursiveFunction]] = None,
        function_call: Optional[str | CursiveFunction] = None,
        on_token: Optional[CursiveAskOnToken] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        frequency_penalty: Optional[int] = None,
        best_of: Optional[int] = None,
        n: Optional[int] = None,
        logit_bias: Optional[dict[str, int]] = None,
        user: Optional[str] = None,
        stream: Optional[bool] = None,
        messages: Optional[list[CompletionMessage]] = None,
        prompt: Optional[str] = None,
    ):
        model = model.value if isinstance(model, CursiveModel) else model
        
        result = build_answer(
            cursive=self,
            model=model,
            system_message=system_message,
            functions=functions,
            function_call=function_call,
            on_token=on_token,
            max_tokens=max_tokens,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            best_of=best_of,
            n=n,
            logit_bias=logit_bias,
            user=user,
            stream=stream,
            messages=messages,
            prompt=prompt,
        )
        if result and result.error:
            return CursiveAnswer[CursiveError](error=result.error)

        new_messages = [
            *(result and result.messages or []),
            CompletionMessage(
                role='assistant',
                content=result and result.answer or '',
            )
        ]

        return CursiveAnswer(
            result=result,
            messages=new_messages,
            cursive=self,
        )
        
        
    def embed(self, content: str):
        options = {
            'model': 'text-embedding-ada-002',
            'input': content,
        }
        self._hooks.call_hook('embedding:before', CursiveHookPayload(data=options))
        start = time.time()
        try:
            data = self._vendor.openai.Embedding.create(
                input=options['input'],
                model='text-embedding-ada-002'
            )
            
            result = {
                'embedding': data['data'][0]['embedding'], # type: ignore
            }
            self._hooks.call_hook('embedding:success', CursiveHookPayload(
                data=result,
                time=time.time() - start,
            ))
            self._hooks.call_hook('embedding:after', CursiveHookPayload(
                data=result, 
                duration=time.time() - start
            ))

            return result['embedding']
        except self._vendor.openai.OpenAIError as e:
            error = CursiveError(
                message=str(e),
                details=e,
                code=CursiveErrorCode.embedding_error
            )
            self._hooks.call_hook('embedding:error', CursiveHookPayload(
                data=error,
                error=error,
                duration=time.time() - start
            ))
            self._hooks.call_hook('embedding:after', CursiveHookPayload(
                data=error,
                error=error,
                duration=time.time() - start
            ))
            raise error
    
        



def resolve_options(
    model: Optional[str | CursiveModel] = None,
    system_message: Optional[str] = None,
    functions: Optional[list[CursiveFunction]] = None,
    function_call: Optional[str | CursiveFunction] = None,
    on_token: Optional[CursiveAskOnToken] = None,
    max_tokens: Optional[int] = None,
    stop: Optional[list[str]] = None,
    temperature: Optional[int] = None,
    top_p: Optional[int] = None,
    presence_penalty: Optional[int] = None,
    frequency_penalty: Optional[int] = None,
    best_of: Optional[int] = None,
    n: Optional[int] = None,
    logit_bias: Optional[dict[str, int]] = None,
    user: Optional[str] = None,
    stream: Optional[bool] = None,
    messages: Optional[list[CompletionMessage]] = None,
    prompt: Optional[str] = None,
):
    functions = functions or []
    messages = messages or []
    model = model or 'gpt-3.5-turbo'

    # TODO: Add support for function call resolving
    vendor = resolve_vendor_from_model(model)
    resolved_system_message = ''
    if  vendor in ['anthropic', 'cohere', 'replicate'] and len(functions) > 0:
        resolved_system_message = (
            (system_message or '')
            + '\n\n'
            + get_function_call_directives(functions)
        )

    query_messages: list[CompletionMessage] = [message for message in [
            resolved_system_message and CompletionMessage(
                role='system',
                content=resolved_system_message
            ),
            *messages,
            prompt and CompletionMessage(role='user', content=prompt),
        ] if message
    ]

    resolved_function_call = (
            { 'name': function_call.function_schema['name'] }
            if isinstance(function_call, CursiveFunction)
            else function_call
        ) if function_call else None

    options = filter_null_values({
        'on_token': on_token,
        'max_tokens': max_tokens,
        'stop': stop,
        'temperature': temperature,
        'top_p': top_p,
        'presence_penalty': presence_penalty,
        'frequency_penalty': frequency_penalty,
        'best_of': best_of,
        'n': n,
        'logit_bias': logit_bias,
        'user': user,
        'stream': stream,
        'model': model,
        'messages': list(
            map(
                lambda message: filter_null_values(dict(message)),
                query_messages
            )
        ),
        'function_call': resolved_function_call,
    })


    payload = CompletionPayload(**options)

    resolved_options = {
        'on_token': on_token,
        'max_tokens': max_tokens,
        'stop': stop,
        'temperature': temperature,
        'top_p': top_p,
        'presence_penalty': presence_penalty,
        'frequency_penalty': frequency_penalty,
        'best_of': best_of,
        'n': n,
        'logit_bias': logit_bias,
        'user': user,
        'stream': stream,
        'model': model,
        'messages': query_messages
    }

    return payload, resolved_options


def create_completion(
    payload: CompletionPayload,
    cursive: Cursive,
    on_token: Optional[CursiveAskOnToken] = None,
) -> CreateChatCompletionResponseExtended:
    cursive._hooks.call_hook('completion:before', CursiveHookPayload(data=payload))
    data = {}
    start = time.time()

    vendor = resolve_vendor_from_model(payload.model)

    # TODO:    Improve the completion creation based on model to vendor matching
    if vendor == 'openai':
        payload.messages = list(
            map(
                lambda message: {
                    k: v for k, v in message.dict().items() if k != 'id' and v is not None
                },
                payload.messages
            )
        )
        resolved_payload = filter_null_values(payload.dict())
        response = cursive._vendor.openai.ChatCompletion.create(
            **resolved_payload
        )
        if payload.stream:
            data = process_openai_stream(
                payload=payload,
                cursive=cursive,
                response=response,
                on_token=on_token,
            )
            content = ''.join(list(map(lambda choice: choice['message']['content'], data['choices'])))
            data['usage']['completion_tokens'] = get_openai_usage(content)
            data['usage']['total_tokens'] = data['usage']['completion_tokens'] + data['usage']['prompt_tokens']
        else:
            data = response
        
        data['cost'] = resolve_pricing(
            vendor='openai',
            usage=CursiveAskUsage(
                completion_tokens=data['usage']['completion_tokens'],
                prompt_tokens=data['usage']['prompt_tokens'],
                total_tokens=data['usage']['total_tokens'],
            ),
            model=data['model']
        )
    elif vendor == 'anthropic':
        response, error = resguard(
            lambda: cursive._vendor.anthropic.create_completion(payload),
            APIError
        )

        if error:
            raise CursiveError(
                message=error.message,
                details=error,
                code=CursiveErrorCode.completion_error
            )
        
        if payload.stream:
            data = process_anthropic_stream(
                payload=payload,
                response=response,
                on_token=on_token,
            )
        else:
            data = {
                'choices': [{ 'message': { 'content': response.completion.lstrip() } }],
                'model': payload.model,
                'id': random_id(),
                'usage': {},
            }

        parse_custom_function_call(data, payload, get_anthropic_usage)

        data['cost'] = resolve_pricing(
            vendor='anthropic',
            usage=CursiveAskUsage(
                completion_tokens=data['usage']['completion_tokens'],
                prompt_tokens=data['usage']['prompt_tokens'],
                total_tokens=data['usage']['total_tokens'],
            ),
            model=data['model']
        )
    elif vendor == 'cohere':
        response, error = cursive._vendor.cohere.create_completion(payload)
        if error:
            raise CursiveError(
                message=error.message,
                details=error,
                code=CursiveErrorCode.completion_error
            )
        if payload.stream:
            # TODO: Implement stream processing for Cohere
            data = process_cohere_stream(
                payload=payload,
                response=response,
                on_token=on_token,
            )
        else:
            data = {
                'choices': [
                    { 'message': { 'content': response.data[0].text.lstrip() } }
                ],
                'model': payload.model,
                'id': random_id(),
                'usage': {},
            }

        parse_custom_function_call(data, payload, get_cohere_usage)

        data['cost'] = resolve_pricing(
            vendor='cohere',
            usage=CursiveAskUsage(
                completion_tokens=data['usage']['completion_tokens'],
                prompt_tokens=data['usage']['prompt_tokens'],
                total_tokens=data['usage']['total_tokens'],
            ),
            model=data['model']
        )
    elif vendor == 'replicate':
        response, error = cursive._vendor.replicate.create_completion(payload)
        if error:
            raise CursiveError(
                message=error,
                details=error,
                code=CursiveErrorCode.completion_error
            )
        # TODO: Implement stream processing for Replicate
        stream_transformer = StreamTransformer(
            on_token=on_token,
            payload=payload,
            response=response,
        )

        def get_current_token(part):
            part.value = part.value

        stream_transformer.on('get_current_token', get_current_token)

        data = stream_transformer.process()

        parse_custom_function_call(data, payload)
    end = time.time()

    if data.get('error'):
        error = CursiveError(
            message=data['error'].message,
            details=data['error'],
            code=CursiveErrorCode.completion_error
        )
        hook_payload = CursiveHookPayload(data=None, error=error, duration=end - start)
        cursive._hooks.call_hook('completion:error', hook_payload)
        cursive._hooks.call_hook('completion:after', hook_payload)
        raise error

    hook_payload = CursiveHookPayload(data=data, error=None, duration=end - start)
    cursive._hooks.call_hook('completion:success', hook_payload)
    cursive._hooks.call_hook('completion:after', hook_payload)
    return CreateChatCompletionResponseExtended(**data)

def ask_model(
    cursive,
    model: Optional[str | CursiveModel] = None,
    system_message: Optional[str] = None,
    functions: Optional[list[CursiveFunction]] = None,
    function_call: Optional[str | CursiveFunction] = None,
    on_token: Optional[CursiveAskOnToken] = None,
    max_tokens: Optional[int] = None,
    stop: Optional[list[str]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    best_of: Optional[int] = None,
    n: Optional[int] = None,
    logit_bias: Optional[dict[str, float]] = None,
    user: Optional[str] = None,
    stream: Optional[bool] = None,
    messages: Optional[list[CompletionMessage]] = None,
    prompt: Optional[str] = None,
) -> CursiveAskModelResponse: 
    payload, resolved_options = resolve_options(
        model=model,
        system_message=system_message,
        functions=functions,
        function_call=function_call,
        on_token=on_token,
        max_tokens=max_tokens,
        stop=stop,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        best_of=best_of,
        n=n,
        logit_bias=logit_bias,
        user=user,
        stream=stream,
        messages=messages,
        prompt=prompt,
    )
    start = time.time()

    functions = functions or []

    if (type(function_call) == CursiveFunction):
        functions.append(function_call)

    function_schemas = list(map(lambda function: function.function_schema, functions))

    if len(function_schemas) > 0:
        payload.functions = function_schemas

    completion, error = resguard(lambda: create_completion(
        payload=payload,
        cursive=cursive,
        on_token=on_token,
    ), CursiveError)

    if error:
        if (not error.details):
            raise CursiveError(
                message=f'Unknown error: {error.message}',
                details=error,
                code=CursiveErrorCode.unknown_error
            ) from error
        try:
            cause = error.details.code or error.details.type
            if (cause == 'context_length_exceeded'):
                if (
                    not cursive.expand
                    or (cursive.expand and cursive.expand.enabled)
                ):
                    default_model = (
                        (
                            cursive.expand
                            and cursive.expand.defaultsTo
                        )
                        or 'gpt-3.5-turbo-16k'
                    )
                    model_mapping = (
                        (
                            cursive.expand
                            and cursive.expand.model_mapping
                        )
                        or {}
                    )
                    resolved_model = model_mapping[model] or default_model
                    completion, error = resguard(
                        lambda: create_completion(
                            payload={ **payload, 'model': resolved_model },
                            cursive=cursive,
                            on_token=on_token,
                        ),
                        CursiveError,
                    )
            elif cause == 'invalid_request_error':
                raise CursiveError(
                    message='Invalid request',
                    details=error.details,
                    code=CursiveErrorCode.invalid_request_error,
                )
        except Exception as e:
            error = CursiveError(
                message=f'Unknown error: {e}',
                details=e,
                code=CursiveErrorCode.unknown_error
            )

        # TODO: Handle other errors
        if error:
            # TODO: Add a more comprehensive retry strategy
            for i in range(cursive.options.max_retries or 0):
                completion, error = resguard(
                    lambda: create_completion(
                        payload=payload,
                        cursive=cursive,
                        on_token=on_token,
                    ),
                    CursiveError
                )

                if error:
                    if i > 3:
                        time.sleep((i - 3) * 2)
                    break

    if error:
        error = CursiveError(
            message='Error while completing request',
            details=error.details,
            code=CursiveErrorCode.completion_error
        )
        hook_payload = CursiveHookPayload(error=error)
        cursive._hooks.call_hook('ask:error', hook_payload)
        cursive._hooks.call_hook('ask:after', hook_payload)
        raise error

    if (
        completion
        and completion.choices
        and len(completion.choices) > 0
        and completion.choices[0].get('message')
        and completion.choices[0]['message'].get('function_call')
    ):

        payload.messages.append({
            'role': 'assistant',
            'function_call': completion.choices[0]['message'].get('function_call'),
            'content': '',
        })
        func_call = completion.choices[0]['message'].get('function_call')
        function_definition = next(
            (f for f in functions if f.function_schema['name'] == func_call['name']),
            None
        )

        if not function_definition:
            return ask_model(**{
                **resolved_options,
                'function_call': 'none',
                'messages': payload.messages,
                'cursive': cursive,
            })

        name = func_call['name']
        called_function = next((func for func in payload.functions if func['name'] == name), None)
        arguments = json.loads(func_call['arguments'] or '{}')
        if called_function:
            props = called_function['parameters']['properties']
            for k, v in props.items():
                if k in arguments:
                    try:
                        arg_type = v['type']
                        if arg_type == 'string':
                            arguments[k] = str(arguments[k])
                        elif arg_type == 'number':
                            arguments[k] = float(arguments[k])
                        elif arg_type == 'integer':
                            arguments[k] = int(arguments[k])
                        elif arg_type == 'boolean':
                            arguments[k] = bool(arguments[k])
                    except Exception:
                        pass

        function_result, error = resguard(
            lambda: function_definition.definition(**arguments),
        )

        if error:
            raise CursiveError(
                message=f'Error while running function ${func_call["name"]}',
                details=error,
                code=CursiveErrorCode.function_call_error,
            )

        messages = payload.messages or []

        messages.append(CompletionMessage(
            role='function',
            name=func_call['name'],
            content=json.dumps(function_result or ''),
        ))

        if function_definition.pause:

            completion.function_result = function_result
            return CursiveAskModelResponse(
                answer=CreateChatCompletionResponseExtended(**completion.dict()),
                messages=messages,
            )
        else:
            return ask_model(**{
                **resolved_options,
                'functions': functions,
                'messages': messages,
                'cursive': cursive,
            })

    end = time.time()
    hook_payload = CursiveHookPayload(data=completion, duration=end - start)
    cursive._hooks.call_hook('ask:after', hook_payload)
    cursive._hooks.call_hook('ask:success', hook_payload)

    return CursiveAskModelResponse(
        answer=completion,
        messages=payload.messages or [],
    )


def build_answer(
    cursive,
    model: Optional[str | CursiveModel] = None,
    system_message: Optional[str] = None,
    functions: Optional[list[CursiveFunction]] = None,
    function_call: Optional[str | CursiveFunction] = None,
    on_token: Optional[CursiveAskOnToken] = None,
    max_tokens: Optional[int] = None,
    stop: Optional[list[str]] = None,
    temperature: Optional[int] = None,
    top_p: Optional[int] = None,
    presence_penalty: Optional[int] = None,
    frequency_penalty: Optional[int] = None,
    best_of: Optional[int] = None,
    n: Optional[int] = None,
    logit_bias: Optional[dict[str, int]] = None,
    user: Optional[str] = None,
    stream: Optional[bool] = None,
    messages: Optional[list[CompletionMessage]] = None,
    prompt: Optional[str] = None,
):
    result, error = resguard(
        lambda: ask_model(
            cursive=cursive,
            model=model,
            system_message=system_message,
            functions=functions,
            function_call=function_call,
            on_token=on_token,
            max_tokens=max_tokens,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            best_of=best_of,
            n=n,
            logit_bias=logit_bias,
            user=user,
            stream=stream,
            messages=messages,
            prompt=prompt,
        ),
        CursiveError
    )

    if error:
        return CursiveEnrichedAnswer(
            error=error,
            usage=None,
            model=model or 'gpt-3.5-turbo',
            id=None,
            choices=None,
            function_result=None,
            answer=None,
            messages=None,
            cost=None,
        )
    else:
        usage = CursiveAskUsage(
            completion_tokens=result.answer.usage['completion_tokens'],
            prompt_tokens=result.answer.usage['prompt_tokens'],
            total_tokens=result.answer.usage['total_tokens'],
        ) if result.answer.usage else None

        return CursiveEnrichedAnswer(
            error=None,
            model=result.answer.model,
            id=result.answer.id,
            usage=usage,
            cost=result.answer.cost,
            choices=list(
                map(lambda choice: choice['message']['content'], result.answer.choices)
            ),
            function_result=result.answer.function_result or None,
            answer=result.answer.choices[-1]['message']['content'],
            messages=result.messages,
        )


    
class CursiveConversation:
    _cursive: Cursive
    messages: list[CompletionMessage]
    
    def __init__(self, messages: list[CompletionMessage]):
        self.messages = messages
        
        
    def ask(
        self,
        model: Optional[str | CursiveModel] = None,
        system_message: Optional[str] = None,
        functions: Optional[list[CursiveFunction]] = None,
        function_call: Optional[str | CursiveFunction] = None,
        on_token: Optional[CursiveAskOnToken] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        frequency_penalty: Optional[int] = None,
        best_of: Optional[int] = None,
        n: Optional[int] = None,
        logit_bias: Optional[dict[str, int]] = None,
        user: Optional[str] = None,
        stream: Optional[bool] = None,
        prompt: Optional[str] = None,
    ):
        messages=[
            *self.messages,
        ]

        result = build_answer(
            cursive=self._cursive,
            model=model,
            system_message=system_message,
            functions=functions,
            function_call=function_call,
            on_token=on_token,
            max_tokens=max_tokens,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            best_of=best_of,
            n=n,
            logit_bias=logit_bias,
            user=user,
            stream=stream,
            messages=messages,
            prompt=prompt,
        )

        if result and result.error:
            return CursiveAnswer[CursiveError](error=result.error)

        new_messages = [
            *(result and result.messages or []),
            CompletionMessage(role='assistant', content=result and result.answer or ''),
        ]

        return CursiveAnswer[None](
            result=result,
            messages=new_messages,
            cursive=self._cursive,
        )


def use_cursive(
    max_retries: Optional[int] = None,
    expand: Optional[CursiveSetupOptionsExpand] = None,
    debug: Optional[bool] = None,
    openai: Optional[dict[str, Any]] = None,
    anthropic: Optional[dict[str, Any]] = None,
):
    return Cursive(
        max_retries=max_retries,
        expand=expand,
        debug=debug,
        openai=openai,
        anthropic=anthropic,
    )


E = TypeVar("E", None, CursiveError)

class CursiveAnswer(Generic[E]):
    choices: Optional[list[str]]
    id: Optional[str]
    model: Optional[str | CursiveModel]
    usage: Optional[CursiveAskUsage]
    cost: Optional[CursiveAskCost]
    error: Optional[E]
    function_result: Optional[Any]
    # The text from the answer of the last choice
    answer: Optional[str]
    # A conversation instance with all the messages so far, including this one
    conversation: Optional[CursiveConversation]

    def __init__(
        self, 
        result: Optional[Any] = None,
        error: Optional[E] = None,
        messages: Optional[list[CompletionMessage]] = None,
        cursive: Optional[Cursive] = None,
    ):
        if error:
            self.error = error
            self.choices = None
            self.id = None
            self.model = None
            self.usage = None
            self.cost = None
            self.answer = None
            self.conversation = None
            self.functionResult = None
        elif result:
            self.error = None
            self.choices = result.choices
            self.id = result.id
            self.model = result.model
            self.usage = result.usage
            self.cost = result.cost
            self.answer = result.answer
            self.function_result = result.function_result
            if messages:
                conversation = CursiveConversation(messages)
                if cursive:
                    conversation._cursive = cursive       
                self.conversation = conversation

    def __str__(self):
        if self.error:
            return f"CursiveAnswer(error={self.error})"
        else:
            return (
                f"CursiveAnswer(\n\tchoices={self.choices}\n\tid={self.id}\n\t"
                f"model={self.model}\n\tusage=(\n\t\t{self.usage}\n\t)\n\tcost=(\n\t\t{self.cost}\n\t)\n\t"
                f"answer={self.answer}\n\tconversation={self.conversation}\n)"
            )

class CursiveVendors(BaseModel):
    openai: Optional[Any] = None
    anthropic: Optional[AnthropicClient] = None
    cohere: Optional[CohereClient] = None
    replicate: Optional[ReplicateClient] = None
