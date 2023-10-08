from typing import List
from cursive.compat.pydantic import BaseModel
from cursive.function import cursive_function
from cursive import Cursive

class Input(BaseModel):
    input: str
    idx: int

@cursive_function(pause=True)
def gen_character_list(inputs: List[Input]):
    """
    Given a prompt (which is directives for a LLM), generate possible inputs that could be fed to it.
    Generate 10 inputs.
    
    inputs: A list of inputs.
    """
    return inputs

c = Cursive()

res = c.ask(
    prompt="Generate a headline for a SaaS company.",
    model="gpt-4",
    function_call=gen_character_list
)

print(res.function_result)