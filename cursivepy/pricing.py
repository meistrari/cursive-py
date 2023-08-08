import json
import os
from typing import Literal

from .custom_types import CursiveAskCost, CursiveAskUsage
from .utils import destructure_items


def resolve_pricing(
    vendor: Literal['openai', 'anthropic'],
    usage: CursiveAskUsage,
    model: str
):
    version: str
    prices: dict[str, dict[str, str]]

    file = open(os.path.abspath(f'./cursive_py/assets/price/{vendor}.json'))
    version, prices = destructure_items(
        keys=["version"],
        dictionary=json.load(file)
    )

    models_available = list(prices.keys())
    model_match = next((m for m in models_available if model.startswith(m)), None)
    
    if not model_match:
        raise Exception(f'Unknown model {model}')
    
    model_price = prices[model_match]    
    completion =  usage.completion_tokens * float(model_price["completion"]) / 1000
    prompt = usage.prompt_tokens * float(model_price["prompt"]) / 1000
    
    cost = CursiveAskCost(
        completion=completion,
        prompt=prompt,
        total=completion + prompt,
        version=version
    )

    return cost
