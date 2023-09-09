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
    def __init__(self, function: Callable, pause=False):
        self.definition = function
        self.description = dedent(function.__doc__ or "")
        self.parameters = validate_arguments(function).model.schema()
        self.pause = pause

        # Delete ['v__duplicate_kwargs', 'args', 'kwargs'] from parameters
        for k in ["v__duplicate_kwargs", "args", "kwargs"]:
            if k in self.parameters["properties"]:
                del self.parameters["properties"][k]

        for k, v in self.parameters["properties"].items():
            # Find the parameter description in the docstring
            match = re.search(rf"{k}: (.*)", self.description)
            if match:
                v["description"] = match.group(1)

        schema = {}
        if self.parameters:
            schema = self.parameters

        self.function_schema = {
            "parameters": {
                "type": schema.get("type"),
                "properties": schema.get("properties") or {},
                "required": schema.get("required") or [],
            },
            "description": self.description,
            "name": self.parameters["title"],
        }

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
