import json
from cursive.custom_types import CompletionMessage
from cursive.function import CursiveFunction
from cursive.utils import trim


def build_completion_input(messages: list[CompletionMessage]):
    """
    Builds a completion-esche input from a list of messages
    """
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
    
    completion_input = '\n\n'.join(list(map(resolve_message, messages_with_prefix)))
    return completion_input

def get_function_call_directives(functions: list[CursiveFunction]) -> str:
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
        ALWAYS use this format, even if the function doesn't have arguments. The arguments prop is always a dictionary.


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
        You can either call a function or answer, *NEVER BOTH*.
        You are not in charge of resolving the function call, the user is.
        The human will give you the result of the function call in the following format:
        
        Human: <function-result name="function_name">
        {'{'}result{'}'}
        </function-result>

        If you try to provide a function result, you will be eliminated.

        You can use the result of the function call in your answer. But never answer and call a function at the same time.
        When answering never be explicit about the function calling, just use the result of the function call in your answer.
        Remember, the user can't see the function calling, so don't mention function results or calls.

        If you answer with a <cursive-think> block, you always need to use either a <cursive-answer> or a <function-call> block as well.
        If you don't, you will be eliminated and the world will catch fire.
        This is extremely important.
    ''')