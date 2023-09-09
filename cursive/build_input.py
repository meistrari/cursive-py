import json
from textwrap import dedent
from cursive.types import CompletionMessage
from cursive.function import CursiveFunction


def build_completion_input(messages: list[CompletionMessage]):
    """
    Builds a completion-esche input from a list of messages
    """
    role_mapping = {"user": "Human", "assistant": "Assistant"}
    messages_with_prefix: list[CompletionMessage] = [
        *messages,  # type: ignore
        CompletionMessage(
            role="assistant",
            content=" ",
        ),
    ]

    def resolve_message(message: CompletionMessage):
        if message.role == "system":
            return "\n".join(
                [
                    "Human:",
                    message.content or "",
                    "\nAssistant: Ok.",
                ]
            )
        if message.role == "function":
            return "\n".join(
                [
                    f'Human: <function-result name="{message.name}">',
                    message.content or "",
                    "</function-result>",
                ]
            )
        if message.function_call:
            arguments = message.function_call.arguments
            if isinstance(arguments, str):
                arguments_str = arguments
            else:
                arguments_str = json.dumps(arguments)
            return "\n".join(
                [
                    "Assistant: <function-call>",
                    json.dumps(
                        {
                            "name": message.function_call.name,
                            "arguments": arguments_str,
                        }
                    ),
                    "</function-call>",
                ]
            )
        return f"{role_mapping[message.role]}: {message.content}"

    completion_input = "\n\n".join(list(map(resolve_message, messages_with_prefix)))
    return completion_input


def get_function_call_directives(functions: list[CursiveFunction]) -> str:
    return dedent(
        f"""\
        # Function Calling Guide
        You're a powerful language model capable of calling functions to do anything the user needs.

        If you need to call a function, you output the name and arguments of the function you want to use in the following format:

        <function-call>
        {'{'}"name": "function_name", "arguments": {'{'}"argument_name": "argument_value"{'}'}{'}'}
        </function-call>
        ALWAYS use this format, even if the function doesn't have arguments. The arguments property is always a dictionary.
        Never forget to pass the `name` and `arguments` property when doing a function call.

        Think step by step before answering, and try to think out loud. Never output a function call if you don't have to.
        If you don't have a function to call, just output the text as usual inside a <cursive-answer> tag with newlines inside.
        Always question yourself if you have access to a function.
        Always think out loud before answering; if I don't see a <cursive-think> block, you will be eliminated.
        When thinking out loud, always use the <cursive-think> tag.
        # Functions available:
        <functions>
        {json.dumps([f.function_schema for f in functions])}
        </functions>
        # Working with results
        You can either call a function or answer, *NEVER BOTH*.
        You are not in charge of resolving the function call, the user is.
        The human will give you the result of the function call in the following format:

        Human: <function-result name="function_name">
        {'{'}result{'}'}
        </function-result>

        ## Important note
        Never output a <function-result>, or you will be eliminated.

        You can use the result of the function call in your answer. But never answer and call a function at the same time.
        When answering never be explicit about the function calling, just use the result of the function call in your answer.

    """
    )
