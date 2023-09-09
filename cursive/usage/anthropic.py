from anthropic import Anthropic

from cursive.build_input import build_completion_input

from ..types import CompletionMessage


def get_anthropic_usage(content: str | list[CompletionMessage]):
    client = Anthropic()

    if type(content) != str:
        content = build_completion_input(content)

    return client.count_tokens(content)
