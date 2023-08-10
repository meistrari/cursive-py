from cursive import Cursive, cursive_function
import argparse
import subprocess

parser = argparse.ArgumentParser(description='Solves your CLI problems')
parser.add_argument('question', type=str, nargs=argparse.REMAINDER, help='Question to ask AI')
args = parser.parse_args()

cursive = Cursive()

@cursive_function(pause=True)
def execute_command(command: str) -> str:
    """
    Executes a CLI command from the user prompt

    command: The command to execute
    """
    return command

res = cursive.ask(
    prompt=' '.join(args.question),
    system_message=
        "You are a CLI assistant, executes commands from the user prompt."
        "You have permission, so just use the function you're provided"
        "Always assume the user wants to run a command on the CLI"
        "Assume they're using a MacOS terminal.",
    functions=[execute_command],
)

conversation = res.conversation

while True:
    if res.function_result:
        print(f'Executing command:\n\t$ {res.function_result}')
        print('Press enter to continue or N/n to cancel')
        
        prompt = input('> ')
        if prompt.lower() == 'n':
            print('Command cancelled')
            exit(0)
        elif prompt == '':
            subprocess.run(res.function_result, shell=True)
            exit(0)
        else:
            res = conversation.ask(
                prompt=prompt,
                functions=[execute_command],
            )
            conversation = res.conversation
    else:
        print(res.answer, end='\n\n')
        prompt = input('> ')
        res = conversation.ask(
            prompt=prompt,
            functions=[execute_command],
        )
        conversation = res.conversation


