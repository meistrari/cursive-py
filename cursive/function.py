import re
from textwrap import dedent
from typing import Any, Callable

from cursive.compat.pydantic import BaseModel, validate_arguments


class CursiveCustomFunction(BaseModel):
    definition: Callable
    description: str = ""
    function_schema: dict[str, Any]
    pause: bool = False

    class Config:
        arbitrary_types_allowed = True


class CursiveFunction(CursiveCustomFunction):
    def __setup__(self, function: Callable):
        definition = function
        description = dedent(function.__doc__ or "").strip()
        parameters = validate_arguments(function).model.schema()


        # Delete ['v__duplicate_kwargs', 'args', 'kwargs'] from parameters
        for k in ["v__duplicate_kwargs", "args", "kwargs"]:
            if k in parameters["properties"]:
                del parameters["properties"][k]

        for k, v in parameters["properties"].items():
            # Find the parameter description in the docstring
            match = re.search(rf"{k}: (.*)", description)
            if match:
                v["description"] = match.group(1)

        schema = {}
        if parameters:
            schema = parameters

        properties = schema.get("properties") or {}
        definitions = schema.get("definitions") or {}
        resolved_properties = remove_key_deep(resolve_ref(properties, definitions), "title")
        

        function_schema = {
            "parameters": {
                "type": schema.get("type"),
                "properties": resolved_properties,
                "required": schema.get("required") or [],
            },
            "description": description,
            "name": parameters["title"],
        }

        return {
            "definition": definition,
            "description": description,
            "function_schema": function_schema,
        }

    def __init__(self, function: Callable, pause=False):
        setup = self.__setup__(function)
        super().__init__(**setup, pause=pause)

    def __call__(self, *args, **kwargs):
        # Validate arguments and parse them
        return self.definition(*args, **kwargs)


def cursive_function(pause=False):
    def decorator(function: Callable = None):
        if function is None:
            return lambda function: CursiveFunction(function, pause=pause)
        else:
            return CursiveFunction(function, pause=pause)

    return decorator

def resolve_ref(data, definitions):
    """
    Recursively checks for a $ref key in a dictionary and replaces it with an entry in the definitions
    dictionary, changing the key `$ref` to `type`.

    Args:
        data (dict): The data dictionary to check for $ref keys.
        definitions (dict): The definitions dictionary to replace $ref keys with.

    Returns:
        dict: The data dictionary with $ref keys replaced.
    """
    if isinstance(data, dict):
        if "$ref" in data:
            ref = data["$ref"].split('/')[-1]
            if ref in definitions:
                definition = definitions[ref]
                data = definition
        else:
            for key, value in data.items():
                data[key] = resolve_ref(value, definitions)
    elif isinstance(data, list):
        for index, value in enumerate(data):
            data[index] = resolve_ref(value, definitions)
    return data

def remove_key_deep(data, key):
    """
    Recursively removes a key from a dictionary.

    Args:
        data (dict): The data dictionary to remove the key from.
        key (str): The key to remove from the dictionary.

    Returns:
        dict: The data dictionary with the key removed.
    """
    if isinstance(data, dict):
        data.pop(key, None)
        for k, v in data.items():
            data[k] = remove_key_deep(v, key)
    elif isinstance(data, list):
        for index, value in enumerate(data):
            data[index] = remove_key_deep(value, key)
    return data