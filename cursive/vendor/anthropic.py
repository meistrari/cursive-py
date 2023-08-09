import json
import re
from typing import Any, Optional

from anthropic import Anthropic

from ..custom_types import (
    CompletionMessage,
    CompletionPayload,
    CursiveAskOnToken,
    CursiveFunction,
)
from ..usage.anthropic import get_anthropic_usage
from ..utils import filter_null_values, random_id, trim


class AnthropicClient:
    client: Anthropic

    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)

    def create_completion(self, payload: CompletionPayload):
        prompt = build_anthropic_input(payload.messages)
        payload = filter_null_values({
            'model': payload.model,
            'max_tokens_to_sample': payload.max_tokens or 100000,
            'prompt': prompt,
            'temperature': payload.temperature or 0.7,
            'top_p': payload.top_p,
            'stop_sequences': payload.stop,
            'stream': payload.stream or False,
        })
        return self.client.completions.create(**payload)



def get_anthropic_function_call_directives(functions: list[CursiveFunction]) -> str:
    return trim(f'''
        # Function Calling Guide
        You're a powerful language model capable of using functions to do anything the user needs.
        
        If you need to use a function, always output the result of the function call using the <function-call> tag using the following format:
        <function-call>
        {'{'}
            "name": "function_name",
            "arguments": {'{'}
                "argument_name": "argument_value"
            {'}'}
        {'}'}
        </function-call>
        Never escape the function call, always output it as it is.


        Think step by step before answering, and try to think out loud. Never output a function call if you don't have to.
        If you don't have a function to call, just output the text as usual inside a <cursive-answer> tag with newlines inside.
        Always question yourself if you have access to a function.
        Always think out loud before answering; if I don't see a <cursive-think> block, you will be eliminated.
        When thinking out loud, always use the <cursive-think> tag.
        # Functions available:
        <functions>
        {json.dumps(list(map(lambda f: f.function_schema, functions)))}
        </functions>

        # Working with results
        You can either call a function or answer, **NEVER BOTH**.
        You are not in charge of resolving the function call, the user is.
        It will give you the result of the function call in the following format:
        
        Human: <function-result name="function_name">
        result
        </function-result>

        You can use the result of the function call in your answer. But never answer and call a function at the same time.
        When answering never be explicit about the function calling, just use the result of the function call in your answer.
        Remember, the user can't see the function calling, so don't mention function results or calls.

        If you answer with a <cursive-think> block, you always need to use either a <cursive-answer> or a <function-call> block as well.
        If you don't, you will be eliminated and the world will catch fire.
        This is extremely important.
    ''')

def build_anthropic_input(messages: list[CompletionMessage]):
    role_mapping = { 'user': 'Human', 'assistant': 'Assistant' }
    messages_with_prefix: list[CompletionMessage] = [
        *messages, # type: ignore
        CompletionMessage(
            role='assistant',
            content=' ',
        ),
    ]
    def resolve_message(message: CompletionMessage):
        if message.role == 'system':
            return '\n'.join([
                'Human:',
                message.content or '',
                '\nAssistant: Ok.',
            ])
        if message.role == 'function':
            return '\n'.join([
                f'Human: <function-result name="{message.name}">',
                message.content or '',
                '</function-result>',
            ])
        if message.function_call:
            arguments = message.function_call.arguments
            if isinstance(arguments, str):
                arguments_str = arguments
            else:
                arguments_str = json.dumps(arguments)
            return '\n'.join([
                'Assistant: <function-call>',
                json.dumps({
                    'name': message.function_call.name,
                    'arguments': arguments_str,
                }),
                '</function-call>',
            ])
        return f'{role_mapping[message.role]}: {message.content}'
    
    return '\n\n'.join(list(map(resolve_message, messages_with_prefix)))


def process_anthropic_stream(
    payload: CompletionPayload,
    cursive: Any,
    response: Any,
    on_token: Optional[CursiveAskOnToken] = None,
):
    data = {
        'choices': [{ 'message': { 'content': '' } }],
        'usage': {
            'completion_tokens': 0,
            'prompt_tokens': get_anthropic_usage(payload.messages),
        },
        'model': payload.model,
    }

    completion = ''
    for slice in response:
        data = {
            **data,
            'id': random_id(),
        }

        # The completion partial will come with a leading whitespace
        completion += slice.completion
        if not data['choices'][0]['message']['content']:
            completion = completion.lstrip()

        # Check if theres any <function-call> tag. The regex should allow for nested tags
        function_call_tag = re.findall(
            r'<function-call>([\s\S]*?)(?=<\/function-call>|$)',
            completion
        )
        function_name = ''
        function_arguments = ''
        if len(function_call_tag) > 0:
            # Remove <function-call> starting and ending tags, even if the ending tag is partial or missing
            function_call = re.sub(
                    r'^\n|\n$',
                    '',
                    re.sub(
                        r'<\/?f?u?n?c?t?i?o?n?-?c?a?l?l?>?',
                        '',
                        function_call_tag[0]
                    ).strip()
                ).strip()
            # Match the function name inside the JSON
            function_name_matches = re.findall(
                r'"name":\s*"(.+)"',
                function_call
            )
            function_name = len(function_name_matches) > 0 and function_name_matches[0]
            function_arguments_matches = re.findall(
                r'"arguments":\s*(\{.+)\}?',
                function_call,
                re.S
            )
            function_arguments = (
                len(function_arguments_matches) > 0 and
                function_arguments_matches[0]
            )
            if function_arguments:
                # If theres unmatches } at the end, remove them
                unmatched_brackets = re.findall(
                    r'(\{|\})',
                    function_arguments
                )
                if len(unmatched_brackets) % 2:
                    function_arguments = re.sub(
                        r'\}$',
                        '',
                        function_arguments.strip()
                    )

                function_arguments = function_arguments.strip()

        cursive_answer_tag = re.findall(
            r'<cursive-answer>([\s\S]*?)(?=<\/cursive-answer>|$)',
            completion
        )
        tagged_answer = ''
        if cursive_answer_tag:
            tagged_answer = re.sub(
                r'<\/?c?u?r?s?i?v?e?-?a?n?s?w?e?r?>?',
                '',
                cursive_answer_tag[0]
            ).lstrip()

        current_token = completion[
            len(data['choices'][0]['message']['content']):
        ]

        data['choices'][0]['message']['content'] += current_token

        if on_token:
            chunk = None

            if payload.functions:
                if function_name:
                    chunk = {
                        'function_call': {},
                        'content': None,
                    }
                    if function_arguments:
                        # Remove all but the current token from the arguments
                        chunk['function_call']['arguments'] = function_arguments
                    else:
                        chunk['function_call'] = {
                            'name': function_name,
                            'arguments': '',
                        }
                elif tagged_answer:
                    # Token is at the end of the tagged answer
                    regex = fr'(.*){current_token.strip()}$'
                    match = re.findall(
                        regex,
                        tagged_answer
                    )
                    if len(match) > 0 and current_token:
                        chunk = {
                            'function_call': None,
                            'content': current_token,
                        }
            else:
                chunk = {
                    'content': current_token,
                }

            if chunk:
                on_token(chunk)

    return data
