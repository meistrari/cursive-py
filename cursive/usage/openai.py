import json
from typing import AbstractSet, Collection, Literal

import tiktoken

from cursive.types import CompletionMessage


def encode(
    text: str,
    allowed_special: AbstractSet[str] | Literal["all"] = set(),
    disallowed_special: Collection[str] | Literal["all"] = "all",
) -> list[int]:
    enc = tiktoken.get_encoding("cl100k_base")

    return enc.encode(
        text, allowed_special=allowed_special, disallowed_special=disallowed_special
    )


def get_openai_usage(content: str | list[CompletionMessage]):
    if type(content) == list:
        tokens = {
            "per_message": 3,
            "per_name": 1,
        }

        token_count = 3
        for message in content:
            token_count += tokens["per_message"]
            for attribute, value in message.dict().items():
                if attribute == "name":
                    token_count += tokens["per_name"]

                if isinstance(value, dict):
                    value = json.dumps(value, separators=(",", ":"))

                if value is None:
                    continue

                token_count += len(encode(value))

        return token_count
    else:
        return len(encode(content))  # type: ignore


def get_token_count_from_functions(functions: list[dict]):
    token_count = 3
    for fn in functions:
        function_tokens = len(encode(fn["name"]))
        function_tokens += len(encode(fn["description"])) if fn["description"] else 0

        if fn["parameters"] and fn["parameters"]["properties"]:
            properties = fn["parameters"]["properties"]
            for key in properties:
                function_tokens += len(encode(key))
                value = properties[key]
                for field in value:
                    if field in ["type", "description"]:
                        function_tokens += 2
                        function_tokens += len(encode(value[field]))
                    elif field == "enum":
                        function_tokens -= 3
                        for enum_value in value[field]:
                            function_tokens += 3
                            function_tokens += len(encode(enum_value))

            function_tokens += 11

        token_count += function_tokens

    token_count += 12
    return token_count
