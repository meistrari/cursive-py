from typing import Any, Optional

from anthropic import Anthropic

from cursive.build_input import build_completion_input
from cursive.stream import StreamTransformer

from ..types import (
    CompletionPayload,
    CursiveAskOnToken,
)
from ..utils import without_nones


class AnthropicClient:
    client: Anthropic

    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)

    def create_completion(self, payload: CompletionPayload):
        prompt = build_completion_input(payload.messages)
        payload = without_nones(
            {
                "model": payload.model,
                "max_tokens_to_sample": payload.max_tokens or 100000,
                "prompt": prompt,
                "temperature": payload.temperature or 0.7,
                "top_p": payload.top_p,
                "stop_sequences": payload.stop,
                "stream": payload.stream or False,
                **(payload.other or {}),
            }
        )
        return self.client.completions.create(**payload)


def process_anthropic_stream(
    payload: CompletionPayload,
    response: Any,
    on_token: Optional[CursiveAskOnToken] = None,
):
    stream_transformer = StreamTransformer(
        on_token=on_token,
        payload=payload,
        response=response,
    )

    def get_current_token(part):
        part.value = part.value.completion

    stream_transformer.on("get_current_token", get_current_token)
    return stream_transformer.process()
