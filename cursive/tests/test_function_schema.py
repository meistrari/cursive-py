from cursive.compat.pydantic import BaseModel
from cursive.function import cursive_function

def test_function_schema_allows_arbitrary_types():
    
    class Character(BaseModel):
        name: str
        age: int

    @cursive_function()
    def gen_arbitrary_type(character: Character):
        """
        A test function.

        character: A character.
        """
        return f"{character.name} is {character.age} years old."

    assert 'description' in gen_arbitrary_type.function_schema