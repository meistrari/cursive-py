import json
import re
from typing import Any, Callable

from pydantic import validate_arguments
from cursive.custom_types import CompletionPayload

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

def parse_custom_function_call(data: dict[str, Any], payload: CompletionPayload, get_usage: Callable = None):
    # We check for function call in the completion
    has_function_call_regex = r'<function-call ?[ˆ>]*>([^<]+)<\/function-call>'
    function_call_matches = re.findall(
        has_function_call_regex,
        data['choices'][0]['message']['content']
    )

    if len(function_call_matches) > 0:
        function_call = json.loads(function_call_matches.pop().strip())
        name = function_call['name']
        arguments = json.dumps(function_call['arguments'])
        data['choices'][0]['message']['function_call'] = {
            'name': name,
            'arguments': arguments,
        }

    # TODO: Implement cohere usage
    if get_usage:
        data['usage']['prompt_tokens'] = get_usage(payload.messages)
        data['usage']['completion_tokens'] = get_usage(data['choices'][0]['message']['content'])
        data['usage']['total_tokens'] = data['usage']['completion_tokens'] + data['usage']['prompt_tokens']
    else:
        data['usage'] = None


    # We check for answers in the completion
    has_answer_regex = r'<cursive-answer>([^<]+)<\/cursive-answer>'
    answer_matches = re.findall(
        has_answer_regex,
        data['choices'][0]['message']['content']
    )
    if len(answer_matches) > 0:
        answer = answer_matches.pop().strip()
        data['choices'][0]['message']['content'] = answer
    
    # As a defensive measure, we check for <cursive-think> tags
    # and remove them
    has_think_regex = r'<cursive-think>([^<]+)<\/cursive-think>'
    think_matches = re.findall(
        has_think_regex,
        data['choices'][0]['message']['content']
    )
    if len(think_matches) > 0:
        data['choices'][0]['message']['content'] = re.sub(
            has_think_regex,
            '',
            data['choices'][0]['message']['content']
        )

    # Strip leading and trailing whitespaces
    data['choices'][0]['message']['content'] = data['choices'][0]['message']['content'].strip()
