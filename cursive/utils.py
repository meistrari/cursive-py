import random
import string
from typing import Any, Callable, Optional, Tuple, Type, TypeVar, overload
import re


def destructure_items(keys: list[str], dictionary: dict):
    items = [
        dictionary[key] for key in keys
    ]
    
    new_dictionary = {
        k:v for k,v in dictionary.items() if k not in keys
    }

    return *items, new_dictionary


def filter_null_values(dictionary: dict):
    return { k:v for k, v in dictionary.items() if v is not None }


def random_id():
    characters = string.ascii_lowercase + string.digits
    random_id = ''.join(random.choice(characters) for _ in range(10))
    return random_id



def trim(content: str) -> str:
    lines = content.split('\n')
    min_indent = float('inf')
    for line in lines:
        indent = re.search(r'\S', line)
        if indent is not None:
            min_indent = min(min_indent, indent.start())
    
    content = ''
    for line in lines:
        content += f"{line[min_indent:]}\n"
    
    return content.strip()


T = TypeVar('T', bound=Exception)

@overload
def resguard(function: Callable) -> Tuple[Any, Exception | None]:
    ...

@overload
def resguard(function: Callable, exception_type: Type[T]) -> Tuple[Any, T | None]:
    ...

def resguard(
    function: Callable,
    exception_type: Optional[Type[T]] = None
) -> Tuple[Any, T | Exception | None]:
    try:
        return function(), None
    except Exception as e:
        if exception_type:
            if isinstance(e, exception_type):
                return None, e
            else:
                raise e
        else:
            return None, e
