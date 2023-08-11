from ..custom_types import CursiveAvailableModels

model_suffix_to_vendor_mapping = {
    'openai': ['gpt-3.5', 'gpt-4'],
    'anthropic': ['claude-instant', 'claude-2'],
    'cohere': ['command']
}

def resolve_vendor_from_model(model: CursiveAvailableModels):
    for vendor, suffixes in model_suffix_to_vendor_mapping.items():
        if len([m for m in suffixes if model.startswith(m)]) > 0:
            return vendor

    return ''

