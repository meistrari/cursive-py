from tokenizers import Tokenizer
from cursive.build_input import build_completion_input
from cursive.types import CompletionMessage


def get_cohere_usage(content: str | list[CompletionMessage]):
    tokenizer = Tokenizer.from_pretrained("Cohere/command-nightly")

    if type(content) != str:
        content = build_completion_input(content)

    return len(tokenizer.encode(content).ids)
