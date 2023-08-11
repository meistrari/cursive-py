from cohere import Client as Cohere
from cursive.build_input import build_completion_input

from cursive.custom_types import CompletionPayload
from cursive.utils import filter_null_values

class CohereClient:
    client: Cohere
    
    def __init__(self, api_key: str):
        self.client = Cohere(api_key=api_key)

    def create_completion(self, payload: CompletionPayload):
        prompt = build_completion_input(payload.messages)
        payload = filter_null_values({ 
            'model': payload.model,
            'max_tokens': payload.max_tokens or 3000,
            'prompt': prompt.rstrip(),
            'temperature': payload.temperature if payload.temperature is not None else 0.7,
            'stop_sequences': payload.stop,
            'stream': payload.stream or False,
            'frequency_penalty': payload.frequency_penalty,
            'p': payload.top_p,
        })
        try:
            response = self.client.generate(
                **payload,
            )
            
            return response, None
        except Exception as e:
            return None, e




