import re
from typing import Any, Callable, Literal, Optional
from cursive.types import CompletionPayload, CursiveAskOnToken
from cursive.hookable import create_hooks
from cursive.utils import random_id


class StreamTransformer:
    def __init__(
        self,
        payload: CompletionPayload,
        response: Any,
        on_token: Optional[CursiveAskOnToken] = None,
    ):
        self._hooks = create_hooks()
        self.payload = payload
        self.response = response
        self.on_token = on_token

    def on(self, event: Literal["get_current_token"], function: Callable):
        self._hooks.hook(event, function)

    def call_output_hook(self, event: str, payload: Any):
        output = HookOutput(payload)
        self._hooks.call_hook(event, output)
        return output.value

    def process(self):
        data = {
            "choices": [{"message": {"content": ""}}],
            "usage": {
                "completion_tokens": 0,
                "prompt_tokens": 0,
            },
            "model": self.payload.model,
            "id": random_id(),
        }
        completion = ""

        for part in self.response:
            # The completion partial will come with a leading whitespace
            current_token = self.call_output_hook("get_current_token", part)
            if not data["choices"][0]["message"]["content"]:
                completion = completion.lstrip()
            completion += current_token
            # Check if theres any <function-call> tag. The regex should allow for nested tags
            function_call_tag = re.findall(
                r"<function-call>([\s\S]*?)(?=<\/function-call>|$)", completion
            )
            function_name = ""
            function_arguments = ""
            if len(function_call_tag) > 0:
                # Remove <function-call> starting and ending tags, even if the ending tag is partial or missing
                function_call = re.sub(
                    r"^\n|\n$",
                    "",
                    re.sub(
                        r"<\/?f?u?n?c?t?i?o?n?-?c?a?l?l?>?", "", function_call_tag[0]
                    ).strip(),
                ).strip()
                # Match the function name inside the JSON
                function_name_matches = re.findall(r'"name":\s*"(.+)"', function_call)
                function_name = (
                    len(function_name_matches) > 0 and function_name_matches[0]
                )
                function_arguments_matches = re.findall(
                    r'"arguments":\s*(\{.+)\}?', function_call, re.S
                )
                function_arguments = (
                    len(function_arguments_matches) > 0
                    and function_arguments_matches[0]
                )
                if function_arguments:
                    # If theres unmatches } at the end, remove them
                    unmatched_brackets = re.findall(r"(\{|\})", function_arguments)
                    if len(unmatched_brackets) % 2:
                        function_arguments = re.sub(
                            r"\}$", "", function_arguments.strip()
                        )

                    function_arguments = function_arguments.strip()

            cursive_answer_tag = re.findall(
                r"<cursive-answer>([\s\S]*?)(?=<\/cursive-answer>|$)", completion
            )
            tagged_answer = ""
            if cursive_answer_tag:
                tagged_answer = re.sub(
                    r"<\/?c?u?r?s?i?v?e?-?a?n?s?w?e?r?>?", "", cursive_answer_tag[0]
                ).lstrip()

            current_token = completion[len(data["choices"][0]["message"]["content"]) :]
            data["choices"][0]["message"]["content"] += current_token

            if self.on_token:
                chunk = None

                if self.payload.functions:
                    if function_name:
                        chunk = {
                            "function_call": {},
                            "content": None,
                        }
                        if function_arguments:
                            # Remove all but the current token from the arguments
                            chunk["function_call"]["arguments"] = function_arguments
                        else:
                            chunk["function_call"] = {
                                "name": function_name,
                                "arguments": "",
                            }
                    elif tagged_answer:
                        # Token is at the end of the tagged answer
                        regex = rf"(.*){current_token.strip()}$"
                        match = re.findall(regex, tagged_answer)
                        if len(match) > 0 and current_token:
                            chunk = {
                                "function_call": None,
                                "content": current_token,
                            }
                else:
                    chunk = {
                        "content": current_token,
                    }

                if chunk:
                    self.on_token(chunk)

        return data


class HookOutput:
    value: Any

    def __init__(self, value: Any = None):
        self.value = value
