vendors_and_model_prefixes = {
    "openai": ["gpt-3.5", "gpt-4"],
    "anthropic": ["claude-instant", "claude-2"],
    "cohere": ["command"],
    "replicate": ["replicate"],
}


def resolve_vendor_from_model(model: str):
    for vendor, prefixes in vendors_and_model_prefixes.items():
        if any(model.startswith(m) for m in prefixes):
            return vendor

    return ""
