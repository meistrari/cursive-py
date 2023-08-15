import re
from typing import Any, Callable

from pydantic import validate_arguments

from cursive.utils import trim

class CursiveFunction:
    def __init__(self, function: Callable, pause=False):
        validate = validate_arguments(function)
        self.parameters = validate.model.schema()
        self.description = trim(function.__doc__)
        self.pause = pause

        # Delete ['v__duplicate_kwargs', 'args', 'kwargs'] from parameters
        for k in ['v__duplicate_kwargs', 'args', 'kwargs']:
            if k in self.parameters['properties']:
                del self.parameters['properties'][k]


        for k, v in self.parameters['properties'].items():
            # Find the parameter description in the docstring
            match = re.search(rf'{k}: (.*)', self.description)
            if match:
                v['description'] = match.group(1)

        schema = {}
        if self.parameters:
            schema = self.parameters
        
        self.function_schema = {
            'parameters': {
                'type': schema.get('type'),
                'properties': schema.get('properties') or {},
                'required': schema.get('required') or [],
            },
            'description': self.description,
            'name': self.parameters['title'],
        }

        self.definition = function

    def __call__(self, *args: Any):
        # Validate arguments and parse them
        return self.function(*args)


def cursive_function(pause=False):
    def decorator(function: Callable = None):
        if function is None:
            return lambda function: CursiveFunction(function, pause=pause)
        else:
            return CursiveFunction(function, pause=pause)
    return decorator
