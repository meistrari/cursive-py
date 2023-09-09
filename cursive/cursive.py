import atexit
import asyncio
import inspect
import json
from time import time, sleep
from typing import Any, Callable, Generic, Optional, TypeVar
from os import environ as env

from anthropic import APIError
import openai as openai_client
import requests

from cursive.build_input import get_function_call_directives
from cursive.compat.pydantic import BaseModel as PydanticBaseModel
from cursive.custom_function_call import parse_custom_function_call
from cursive.function import CursiveCustomFunction, CursiveFunction
from cursive.model import CursiveModel
from cursive.stream import StreamTransformer
from cursive.usage.cohere import get_cohere_usage
from cursive.vendor.cohere import CohereClient, process_cohere_stream
from cursive.types import (
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
    CursiveHook,
    CursiveHookPayload,
    CursiveSetupOptions,
    CursiveSetupOptionsExpand,
    CursiveLanguageModel,
)
from cursive.hookable import create_debugger, create_hooks
from cursive.pricing import resolve_pricing
from cursive.usage.anthropic import get_anthropic_usage
from cursive.usage.openai import get_openai_usage
from cursive.utils import delete_keys_from_dict, without_nones, random_id, resguard
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
        openrouter: Optional[dict[str, Any]] = None,
    ):
        self._hooks = create_hooks()
        self.options = CursiveSetupOptions(
            max_retries=max_retries,
            expand=expand,
            debug=debug,
        )
        if debug:
            self._debugger = create_debugger(self._hooks, {"tag": "cursive"})

        openai_client.api_key = (openai or {}).get("api_key") or env.get(
            "OPENAI_API_KEY"
        )
        anthropic_client = AnthropicClient(
            (anthropic or {}).get("api_key") or env.get("ANTHROPIC_API_KEY")
        )
        cohere_client = CohereClient(
            (cohere or {}).get("api_key") or env.get("CO_API_KEY", "---")
        )
        replicate_client = ReplicateClient(
            (replicate or {}).get("api_key") or env.get("REPLICATE_API_TOKEN", "---")
        )

        openrouter_api_key = (openrouter or {}).get("api_key") or env.get(
            "OPENROUTER_API_KEY"
        )

        if openrouter_api_key:
            openai_client.api_base = "https://openrouter.ai/api/v1"
            openai_client.api_key = openrouter_api_key
            self.options.is_using_openrouter = True
            session = requests.Session()
            session.headers.update(
                {
                    "HTTP-Referer": openrouter.get(
                        "app_url", "https://cursive.meistrari.com"
                    ),
                    "X-Title": openrouter.get("app_title", "Cursive"),
                }
            )
            openai_client.requestssession = session
            atexit.register(session.close)

        self._vendor = CursiveVendors(
            openai=openai_client,
            anthropic=anthropic_client,
            cohere=cohere_client,
            replicate=replicate_client,
        )

    def on(self, event: CursiveHook, callback: Callable):
        self._hooks.hook(event, callback)

    def ask(
        self,
        model: Optional[str | CursiveLanguageModel] = None,
        system_message: Optional[str] = None,
        functions: Optional[list[Callable]] = None,
        function_call: Optional[str | Callable] = None,
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
        model = model.value if isinstance(model, CursiveLanguageModel) else model

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

        return CursiveAnswer(
            result=result,
            messages=result.messages,
            cursive=self,
        )

    def embed(self, content: str):
        options = {
            "model": "text-embedding-ada-002",
            "input": content,
        }
        self._hooks.call_hook("embedding:before", CursiveHookPayload(data=options))
        start = time()
        try:
            data = self._vendor.openai.Embedding.create(
                input=options["input"], model="text-embedding-ada-002"
            )

            result = {
                "embedding": data["data"][0]["embedding"],  # type: ignore
            }
            self._hooks.call_hook(
                "embedding:success",
                CursiveHookPayload(
                    data=result,
                    time=time() - start,
                ),
            )
            self._hooks.call_hook(
                "embedding:after",
                CursiveHookPayload(data=result, duration=time() - start),
            )

            return result["embedding"]
        except self._vendor.openai.OpenAIError as e:
            error = CursiveError(
                message=str(e), details=e, code=CursiveErrorCode.embedding_error
            )
            self._hooks.call_hook(
                "embedding:error",
                CursiveHookPayload(data=error, error=error, duration=time() - start),
            )
            self._hooks.call_hook(
                "embedding:after",
                CursiveHookPayload(data=error, error=error, duration=time() - start),
            )
            raise error


def resolve_options(
    model: Optional[str | CursiveLanguageModel] = None,
    system_message: Optional[str] = None,
    functions: Optional[list[Callable]] = None,
    function_call: Optional[str | Callable] = None,
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
    cursive: Cursive = None,
):
    messages = messages or []

    functions = functions or []
    functions = [cursive_wrapper(f) for f in functions]

    function_call = cursive_wrapper(function_call)
    if function_call:
        functions.append(function_call)

    # Resolve default model
    model = model or (
        "openai/gpt-3.5-turbo"
        if cursive.options.is_using_openrouter
        else "gpt-3.5-turbo"
    )

    # TODO: Add support for function call resolving
    vendor = (
        "openrouter"
        if cursive.options.is_using_openrouter
        else resolve_vendor_from_model(model)
    )

    resolved_system_message = ""

    if vendor in ["anthropic", "cohere", "replicate"] and len(functions) > 0:
        resolved_system_message = (
            (system_message or "") + "\n\n" + get_function_call_directives(functions)
        )

    query_messages: list[CompletionMessage] = [
        message
        for message in [
            resolved_system_message
            and CompletionMessage(role="system", content=resolved_system_message),
            *messages,
            prompt and CompletionMessage(role="user", content=prompt),
        ]
        if message
    ]

    payload_params = without_nones(
        {
            "on_token": on_token,
            "max_tokens": max_tokens,
            "stop": stop,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "best_of": best_of,
            "n": n,
            "logit_bias": logit_bias,
            "user": user,
            "stream": stream,
            "model": model,
            "messages": [without_nones(dict(m)) for m in query_messages],
        }
    )
    if function_call:
        payload_params["function_call"] = (
            {"name": function_call.function_schema["name"]}
            if isinstance(function_call, CursiveFunction)
            else function_call
        )
    if functions:
        payload_params["functions"] = [fn.function_schema for fn in functions]

    payload = CompletionPayload(**payload_params)

    resolved_options = {
        "on_token": on_token,
        "max_tokens": max_tokens,
        "stop": stop,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "best_of": best_of,
        "n": n,
        "logit_bias": logit_bias,
        "user": user,
        "stream": stream,
        "model": model,
        "messages": query_messages,
        "functions": functions,
    }

    return payload, resolved_options


def create_completion(
    payload: CompletionPayload,
    cursive: Cursive,
    on_token: Optional[CursiveAskOnToken] = None,
) -> CreateChatCompletionResponseExtended:
    cursive._hooks.call_hook("completion:before", CursiveHookPayload(data=payload))
    data = {}
    start = time()

    vendor = (
        "openrouter"
        if cursive.options.is_using_openrouter
        else resolve_vendor_from_model(payload.model)
    )

    # TODO:    Improve the completion creation based on model to vendor matching
    if vendor == "openai" or vendor == "openrouter":
        resolved_payload = without_nones(payload.dict())

        # Remove the ID from the messages before sending to OpenAI
        resolved_payload["messages"] = [
            without_nones(delete_keys_from_dict(message, ["id", "model_config"]))
            for message in resolved_payload["messages"]
        ]

        response = cursive._vendor.openai.ChatCompletion.create(**resolved_payload)
        if payload.stream:
            data = process_openai_stream(
                payload=payload,
                cursive=cursive,
                response=response,
                on_token=on_token,
            )
            content = "".join(
                choice["message"]["content"] for choice in data["choices"]
            )
            data["usage"]["completion_tokens"] = get_openai_usage(content)
            data["usage"]["total_tokens"] = (
                data["usage"]["completion_tokens"] + data["usage"]["prompt_tokens"]
            )
        else:
            data = response

        # If the user is using OpenRouter, there's no usage data
        if usage := data.get("usage"):
            data["cost"] = resolve_pricing(
                vendor="openai",
                model=data["model"],
                usage=CursiveAskUsage(
                    completion_tokens=usage["completion_tokens"],
                    prompt_tokens=usage["prompt_tokens"],
                    total_tokens=usage["total_tokens"],
                ),
            )

    elif vendor == "anthropic":
        response, error = resguard(
            lambda: cursive._vendor.anthropic.create_completion(payload), APIError
        )

        if error:
            raise CursiveError(
                message=error.message,
                details=error,
                code=CursiveErrorCode.completion_error,
            )

        if payload.stream:
            data = process_anthropic_stream(
                payload=payload,
                response=response,
                on_token=on_token,
            )
        else:
            data = {
                "choices": [{"message": {"content": response.completion.lstrip()}}],
                "model": payload.model,
                "id": random_id(),
                "usage": {},
            }

        parse_custom_function_call(data, payload, get_anthropic_usage)

        data["cost"] = resolve_pricing(
            vendor="anthropic",
            usage=CursiveAskUsage(
                completion_tokens=data["usage"]["completion_tokens"],
                prompt_tokens=data["usage"]["prompt_tokens"],
                total_tokens=data["usage"]["total_tokens"],
            ),
            model=data["model"],
        )
    elif vendor == "cohere":
        response, error = cursive._vendor.cohere.create_completion(payload)
        if error:
            raise CursiveError(
                message=error.message,
                details=error,
                code=CursiveErrorCode.completion_error,
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
                "choices": [{"message": {"content": response.data[0].text.lstrip()}}],
                "model": payload.model,
                "id": random_id(),
                "usage": {},
            }

        parse_custom_function_call(data, payload, get_cohere_usage)

        data["cost"] = resolve_pricing(
            vendor="cohere",
            usage=CursiveAskUsage(
                completion_tokens=data["usage"]["completion_tokens"],
                prompt_tokens=data["usage"]["prompt_tokens"],
                total_tokens=data["usage"]["total_tokens"],
            ),
            model=data["model"],
        )
    elif vendor == "replicate":
        response, error = cursive._vendor.replicate.create_completion(payload)
        if error:
            raise CursiveError(
                message=error, details=error, code=CursiveErrorCode.completion_error
            )
        # TODO: Implement stream processing for Replicate
        stream_transformer = StreamTransformer(
            on_token=on_token,
            payload=payload,
            response=response,
        )

        def get_current_token(part):
            part.value = part.value

        stream_transformer.on("get_current_token", get_current_token)

        data = stream_transformer.process()

        parse_custom_function_call(data, payload)
    else:
        raise CursiveError(
            message="Unknown model",
            details=None,
            code=CursiveErrorCode.completion_error,
        )
    end = time()

    if data.get("error"):
        error = CursiveError(
            message=data["error"].message,
            details=data["error"],
            code=CursiveErrorCode.completion_error,
        )
        hook_payload = CursiveHookPayload(data=None, error=error, duration=end - start)
        cursive._hooks.call_hook("completion:error", hook_payload)
        cursive._hooks.call_hook("completion:after", hook_payload)
        raise error

    hook_payload = CursiveHookPayload(data=data, error=None, duration=end - start)
    cursive._hooks.call_hook("completion:success", hook_payload)
    cursive._hooks.call_hook("completion:after", hook_payload)
    return CreateChatCompletionResponseExtended(**data)


def cursive_wrapper(fn):
    if fn is None:
        return None
    elif issubclass(type(fn), CursiveCustomFunction):
        return fn
    elif inspect.isclass(fn) and issubclass(fn, PydanticBaseModel):
        return CursiveModel(fn)
    elif inspect.isfunction(fn):
        return CursiveFunction(fn, pause=True)


def ask_model(
    cursive,
    model: Optional[str | CursiveLanguageModel] = None,
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
        cursive=cursive,
    )
    start = time()

    completion, error = resguard(
        lambda: create_completion(
            payload=payload,
            cursive=cursive,
            on_token=on_token,
        ),
        CursiveError,
    )

    if error:
        if not error.details:
            raise CursiveError(
                message=f"Unknown error: {error.message}",
                details=error,
                code=CursiveErrorCode.unknown_error,
            ) from error
        try:
            cause = error.details.code or error.details.type
            if cause == "context_length_exceeded":
                if not cursive.expand or (cursive.expand and cursive.expand.enabled):
                    default_model = (
                        cursive.expand and cursive.expand.defaultsTo
                    ) or "gpt-3.5-turbo-16k"
                    model_mapping = (
                        cursive.expand and cursive.expand.model_mapping
                    ) or {}
                    resolved_model = model_mapping[model] or default_model
                    completion, error = resguard(
                        lambda: create_completion(
                            payload={**payload, "model": resolved_model},
                            cursive=cursive,
                            on_token=on_token,
                        ),
                        CursiveError,
                    )
            elif cause == "invalid_request_error":
                raise CursiveError(
                    message="Invalid request",
                    details=error.details,
                    code=CursiveErrorCode.invalid_request_error,
                )
        except Exception as e:
            error = CursiveError(
                message=f"Unknown error: {e}",
                details=e,
                code=CursiveErrorCode.unknown_error,
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
                    CursiveError,
                )

                if error:
                    if i > 3:
                        sleep((i - 3) * 2)
                    break

    if error:
        error = CursiveError(
            message="Error while completing request",
            details=error.details,
            code=CursiveErrorCode.completion_error,
        )
        hook_payload = CursiveHookPayload(error=error)
        cursive._hooks.call_hook("ask:error", hook_payload)
        cursive._hooks.call_hook("ask:after", hook_payload)
        raise error

    if (
        completion
        and completion.choices
        and (fn_call := completion.choices[0].get("message", {}).get("function_call"))
    ):
        function: CursiveFunction = next(
            (
                f
                for f in resolved_options["functions"]
                if f.function_schema["name"] == fn_call["name"]
            ),
            None,
        )

        if function is None:
            return ask_model(
                **{
                    **resolved_options,
                    "function_call": None,
                    "messages": payload.messages,
                    "cursive": cursive,
                }
            )

        called_function = function.function_schema
        arguments = json.loads(fn_call["arguments"] or "{}")
        props = called_function["parameters"]["properties"]
        for k, v in props.items():
            if k in arguments:
                try:
                    match v["type"]:
                        case "string":
                            arguments[k] = str(arguments[k])
                        case "number":
                            arguments[k] = float(arguments[k])
                        case "integer":
                            arguments[k] = int(arguments[k])
                        case "boolean":
                            arguments[k] = bool(arguments[k])
                except Exception:
                    pass

        is_async = inspect.iscoroutinefunction(function.definition)
        function_result = None
        try:
            if is_async:
                function_result = asyncio.run(function.definition(**arguments))
            else:
                function_result = function.definition(**arguments)
        except Exception as error:
            raise CursiveError(
                message=f'Error while running function ${fn_call["name"]}',
                details=error,
                code=CursiveErrorCode.function_call_error,
            )

        messages = payload.messages or []
        messages.append(
            CompletionMessage(
                role="assistant",
                name=fn_call["name"],
                content=json.dumps(fn_call),
                function_call=fn_call,
            )
        )

        if function.pause:
            completion.function_result = function_result
            return CursiveAskModelResponse(
                answer=completion,
                messages=messages,
            )
        else:
            return ask_model(
                **{
                    **resolved_options,
                    "functions": functions,
                    "messages": messages,
                    "cursive": cursive,
                }
            )

    end = time()
    hook_payload = CursiveHookPayload(data=completion, duration=end - start)
    cursive._hooks.call_hook("ask:after", hook_payload)
    cursive._hooks.call_hook("ask:success", hook_payload)

    messages = payload.messages or []
    messages.append(
        CompletionMessage(
            role="assistant",
            content=completion.choices[0]["message"]["content"],
        )
    )

    return CursiveAskModelResponse(answer=completion, messages=messages)


def build_answer(
    cursive,
    model: Optional[str | CursiveLanguageModel] = None,
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
        CursiveError,
    )

    if error:
        return CursiveEnrichedAnswer(
            error=error,
            usage=None,
            model=model or "gpt-3.5-turbo",
            id=None,
            choices=None,
            function_result=None,
            answer=None,
            messages=None,
            cost=None,
        )
    else:
        usage = (
            CursiveAskUsage(
                completion_tokens=result.answer.usage["completion_tokens"],
                prompt_tokens=result.answer.usage["prompt_tokens"],
                total_tokens=result.answer.usage["total_tokens"],
            )
            if result.answer.usage
            else None
        )

        return CursiveEnrichedAnswer(
            error=None,
            model=result.answer.model,
            id=result.answer.id,
            usage=usage,
            cost=result.answer.cost,
            choices=[choice["message"]["content"] for choice in result.answer.choices],
            function_result=result.answer.function_result or None,
            answer=result.answer.choices[-1]["message"]["content"],
            messages=result.messages,
        )


class CursiveConversation:
    _cursive: Cursive
    messages: list[CompletionMessage]

    def __init__(self, messages: list[CompletionMessage]):
        self.messages = messages

    def ask(
        self,
        model: Optional[str | CursiveLanguageModel] = None,
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
        messages = [
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

        return CursiveAnswer[None](
            result=result,
            messages=result.messages,
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
    model: Optional[str | CursiveLanguageModel]
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
