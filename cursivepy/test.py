from pydantic import BaseModel, Field

from cursive_py.function import create_function

from .cursive import use_cursive


class Params(BaseModel):
    a: int = Field(..., description="First number")
    b: int = Field(..., description="Second number")
    
def add(a: int, b: int) -> int:
    return a + b


def main():
    cursive = use_cursive(
        openai_options={ 'api_key': '' }
    )

    answer = cursive.ask(
        prompt='Add numbers 3 and 4',
        functions=[
            create_function(
                name='add',
                description='Add two numbers together',
                execute=lambda a,b: a+b,
                parameters=Params
            )
        ]
    )

    for k, v in answer.__dict__.items():
        print(f'{k}: {v}')
    
if __name__ == '__main__':
    main()