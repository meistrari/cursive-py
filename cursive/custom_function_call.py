import json
import re
from typing import Any, Callable
from cursive.types import CompletionPayload


def parse_custom_function_call(
    data: dict[str, Any], payload: CompletionPayload, get_usage: Callable = None
):
    # We check for function call in the completion
    has_function_call_regex = r"<function-call ?[ˆ>]*>([^<]+)<\/function-call>"
    function_call_matches = re.findall(
        has_function_call_regex, data["choices"][0]["message"]["content"]
    )

    if len(function_call_matches) > 0:
        function_call = json.loads(function_call_matches.pop().strip())
        name = function_call["name"]
        arguments = json.dumps(function_call["arguments"])
        data["choices"][0]["message"]["function_call"] = {
            "name": name,
            "arguments": arguments,
        }

    # TODO: Implement cohere usage
    if get_usage:
        data["usage"]["prompt_tokens"] = get_usage(payload.messages)
        data["usage"]["completion_tokens"] = get_usage(
            data["choices"][0]["message"]["content"]
        )
        data["usage"]["total_tokens"] = (
            data["usage"]["completion_tokens"] + data["usage"]["prompt_tokens"]
        )
    else:
        data["usage"] = None

    # We check for answers in the completion
    has_answer_regex = r"<cursive-answer>([^<]+)<\/cursive-answer>"
    answer_matches = re.findall(
        has_answer_regex, data["choices"][0]["message"]["content"]
    )
    if len(answer_matches) > 0:
        answer = answer_matches.pop().strip()
        data["choices"][0]["message"]["content"] = answer

    # As a defensive measure, we check for <cursive-think> tags
    # and remove them
    has_think_regex = r"<cursive-think>([^<]+)<\/cursive-think>"
    think_matches = re.findall(
        has_think_regex, data["choices"][0]["message"]["content"]
    )
    if len(think_matches) > 0:
        data["choices"][0]["message"]["content"] = re.sub(
            has_think_regex, "", data["choices"][0]["message"]["content"]
        )

    # Strip leading and trailing whitespaces
    data["choices"][0]["message"]["content"] = data["choices"][0]["message"][
        "content"
    ].strip()
