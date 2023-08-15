
model_suffix_to_vendor_mapping = {
    'openai': ['gpt-3.5', 'gpt-4'],
    'anthropic': ['claude-instant', 'claude-2'],
    'cohere': ['command'],
    'replicate': ['replicate']
}

def resolve_vendor_from_model(model: str):
    for vendor, suffixes in model_suffix_to_vendor_mapping.items():
        if len([m for m in suffixes if model.startswith(m)]) > 0:
            return vendor

    return ''

