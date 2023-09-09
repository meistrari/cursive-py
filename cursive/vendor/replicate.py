import replicate

from cursive.build_input import build_completion_input
from cursive.types import CompletionPayload
from cursive.utils import without_nones


class ReplicateClient:
    client: replicate.Client

    def __init__(self, api_key: str):
        self.client = replicate.Client(api_key)

    def create_completion(self, payload: CompletionPayload):  # noqa: F821
        prompt = build_completion_input(payload.messages)
        # Resolve model ID from `replicate/<model>`
        version = payload.model[payload.model.find("/") + 1 :]
        resolved_payload = without_nones(
            {
                "max_new_tokens": payload.max_tokens or 2000,
                "max_length": payload.max_tokens or 2000,
                "prompt": prompt,
                "temperature": payload.temperature or 0.7,
                "top_p": payload.top_p,
                "stop": payload.stop,
                "model": version,
                "stream": bool(payload.stream),
                **(payload.other or {}),
            }
        )
        try:
            response = self.client.run(
                version,
                input=resolved_payload,
            )

            return response, None
        except Exception as e:
            print("e", e)
            return None, e
