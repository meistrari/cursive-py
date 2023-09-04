from typing import Literal

from .types import CursiveAskCost, CursiveAskUsage
from .utils import destructure_items

from .assets.price.anthropic import ANTHROPIC_PRICING
from .assets.price.openai import OPENAI_PRICING
from .assets.price.cohere import COHERE_PRICING

VENDOR_PRICING = {
    "openai": OPENAI_PRICING,
    "anthropic": ANTHROPIC_PRICING,
    "cohere": COHERE_PRICING,
}


def resolve_pricing(
    vendor: Literal["openai", "anthropic"], usage: CursiveAskUsage, model: str
):
    if "/" in model:
        vendor, model = model.split("/")

    version: str
    prices: dict[str, dict[str, str]]

    version, prices = destructure_items(
        keys=["version"], dictionary=VENDOR_PRICING[vendor]
    )

    models_available = list(prices.keys())
    model_match = next((m for m in models_available if model.startswith(m)), None)

    if not model_match:
        raise Exception(f"Unknown model {model}")

    model_price = prices[model_match]
    completion = usage.completion_tokens * float(model_price["completion"]) / 1000
    prompt = usage.prompt_tokens * float(model_price["prompt"]) / 1000

    cost = CursiveAskCost(
        completion=completion, prompt=prompt, total=completion + prompt, version=version
    )

    return cost
