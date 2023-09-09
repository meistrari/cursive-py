from cursive.compat.pydantic import BaseModel
from cursive.function import CursiveFunction


class CursiveModel(CursiveFunction):
    def __init__(self, model: type[BaseModel]):
        super().__init__(model, pause=True)


def cursive_model():
    def decorator(model: type[BaseModel] = None):
        if model is None:
            return lambda function: CursiveModel(function)
        else:
            return CursiveModel(model)

    return decorator
