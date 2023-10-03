import re
from textwrap import dedent
from typing import Any, Callable

from cursive.compat.pydantic import BaseModel, validate_arguments


class CursiveCustomFunction(BaseModel):
    definition: Callable
    description: str = ""
    function_schema: dict[str, Any]
    pause: bool = False


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

        function_schema = {
            "parameters": {
                "type": schema.get("type"),
                "properties": schema.get("properties") or {},
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
