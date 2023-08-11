from anthropic import Anthropic

from ..custom_types import CompletionMessage

def get_anthropic_usage(content: str | list[CompletionMessage]):
    client = Anthropic()

    if type(content) == str:
        return client.count_tokens(content)
    
    def function(message: CompletionMessage):
        if message.role == 'system':
            return f'''
                Human: {message.content}
                
                Assistant: Ok.
            '''
        return f'{message.role}: {message.content}'

    mapped_content = '\n\n'.join(list(map(function, content))) # type: ignore

    return client.count_tokens(mapped_content)

