import random
import string

from typing import Any, Callable, Optional, Tuple, Type, TypeVar, overload


def destructure_items(keys: list[str], dictionary: dict):
    items = [dictionary[key] for key in keys]

    new_dictionary = {k: v for k, v in dictionary.items() if k not in keys}

    return *items, new_dictionary


def without_nones(dictionary: dict):
    return {k: v for k, v in dictionary.items() if v is not None}


def random_id():
    characters = string.ascii_lowercase + string.digits
    random_id = "".join(random.choice(characters) for _ in range(10))
    return random_id


def delete_keys_from_dict(dictionary: dict, keys: list[str]):
    return {k: v for k, v in dictionary.items() if k not in set(keys)}


T = TypeVar("T", bound=Exception)


@overload
def resguard(function: Callable) -> Tuple[Any, Exception | None]:
    ...


@overload
def resguard(function: Callable, exception_type: Type[T]) -> Tuple[Any, T | None]:
    ...


def resguard(
    function: Callable, exception_type: Optional[Type[T]] = None
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
