from typing import Any, Optional
from cohere import Client as Cohere
from cursive.build_input import build_completion_input

from cursive.types import CompletionPayload, CursiveAskOnToken
from cursive.stream import StreamTransformer
from cursive.utils import without_nones


class CohereClient:
    client: Cohere

    def __init__(self, api_key: str):
        self.client = Cohere(api_key=api_key)

    def create_completion(self, payload: CompletionPayload):
        prompt = build_completion_input(payload.messages)
        payload = without_nones(
            {
                "model": payload.model,
                "max_tokens": payload.max_tokens or 3000,
                "prompt": prompt.rstrip(),
                "temperature": payload.temperature
                if payload.temperature is not None
                else 0.7,
                "stop_sequences": payload.stop,
                "stream": payload.stream or False,
                "frequency_penalty": payload.frequency_penalty,
                "p": payload.top_p,
            }
        )
        try:
            response = self.client.generate(
                **payload,
            )

            return response, None
        except Exception as e:
            return None, e


def process_cohere_stream(
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
        part.value = part.value.text

    stream_transformer.on("get_current_token", get_current_token)

    return stream_transformer.process()
