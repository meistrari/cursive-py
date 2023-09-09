from typing import Any, Optional

from ..types import CursiveAskOnToken
from ..usage.openai import get_openai_usage, get_token_count_from_functions


def process_openai_stream(
    payload: Any,
    cursive: Any,
    response: Any,
    on_token: Optional[CursiveAskOnToken] = None,
):
    data = {
        "choices": [],
        "usage": {
            "completion_tokens": 0,
            "prompt_tokens": get_openai_usage(payload.messages),
        },
        "model": payload.model,
    }

    if payload.functions:
        data["usage"]["prompt_tokens"] += get_token_count_from_functions(
            payload.functions
        )

    for slice in response:
        data = {
            **data,
            "id": slice["id"],
        }

        for i in range(len(slice["choices"])):
            delta = slice["choices"][i]["delta"]

            if len(data["choices"]) <= i:
                data["choices"].append(
                    {
                        "message": {
                            "function_call": None,
                            "role": "assistant",
                            "content": "",
                        },
                    }
                )

            if (
                delta
                and delta.get("function_call")
                and delta["function_call"].get("name")
            ):
                data["choices"][i]["message"]["function_call"] = delta["function_call"]

            if (
                delta
                and delta.get("function_call")
                and delta["function_call"].get("arguments")
            ):
                data["choices"][i]["message"]["function_call"]["arguments"] += delta[
                    "function_call"
                ]["arguments"]

            if delta and delta.get("content"):
                data["choices"][i]["message"]["content"] += delta["content"]

            if on_token:
                chunk = None
                if delta and delta.get("function_call"):
                    chunk = {
                        "function_call": {
                            k: v for k, v in delta["function_call"].items()
                        },
                        "content": None,
                    }

                if delta and delta.get("content"):
                    chunk = {"content": delta["content"], "function_call": None}

                if chunk:
                    on_token(chunk)

    return data
