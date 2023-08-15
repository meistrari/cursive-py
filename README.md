![Logo](/docs/logo-dark.svg#gh-dark-mode-only)
![Logo](/docs/logo-light.svg#gh-light-mode-only)

Cursive is a universal and intuitive framework for interacting with LLMs.

It works in any JavaScript runtime and has a heavy focus on extensibility and developer experience.

## highlights
<img width=14 height=0 src=""/>✦ **Extensible** - You can easily hook into any part of a completion life cycle. Be it to log, cache, or modify the results.

<img width=14 height=0 src=""/>✦ **Functions** - Easily describe functions that the LLM can use along with its definition, with any model (currently supporting GPT-4, GPT-3.5, Claude 2, and Claude Instant)

<img width=14 height=0 src=""/>✦ **Universal** - Cursive's goal is to bridge as many capabilities between different models as possible. Ultimately, this means that with a single interface, you can allow your users to choose any model.

<img width=14 height=0 src=""/>✦ **Informative** - Cursive comes with built-in token usage and costs calculations, as accurate as possible.

<img width=14 height=0 src=""/>✦ **Reliable** - Cursive comes with automatic retry and model expanding upon exceeding context length. Which you can always configure.

## quickstart
1. Install.

```bash
poetry add cursivepy
# or
pip install cursivepy
```

2. Start using.

```python
from cursive import Cursive

cursive = Cursive()

response = cursive.ask(
    prompt='What is the meaning of life?',
)

print(response.answer)
```

## usage
### Conversation
Chaining a conversation is easy with `cursive`. You can pass any of the options you're used to with OpenAI's API.

```python
res_a = cursive.ask(
    prompt='Give me a good name for a gecko.',
    model='gpt-4',
    max_tokens=16,
)

print(res_a.answer) # Zephyr

res_b = res_b.conversation.ask(
    prompt='How would you say it in Portuguese?'
)

print(res_b.answer) # Zéfiro
```
### Streaming
Streaming is also supported, and we also keep track of the tokens for you!
```python
result = cursive.ask(
    prompt='Count to 10',
    stream=True,
    on_token=lambda partial: print(partial['content'])
)

print(result.usage.total_tokens) # 40
```

### Functions
You can use very easily to define and describe functions, along side with their execution code.
```python
from cursive import cursive_function, Cursive

cursive = Cursive()

@cursive_function()
def add(a: float, b: float):
    """
    Adds two numbers.
    
    a: The first number.
    b: The second number.
    """
    return a + b

res = cursive.ask(
    prompt='What is the sum of 232 and 243?',
    functions=[sum],
)

print(res.answer) # The sum of 232 and 243 is 475.
```

The functions' result will automatically be fed into the conversation and another completion will be made. If you want to prevent this, you can add `pause` to your function definition.

```python

@cursive_function(pause=True)
def create_character(name: str, age: str):
    """
    Creates a character.
    
    name: The name of the character.
    age: The age of the character.
    """
    return {
        'name': name,
        'age': age,
    }

res = cursive.ask(
    prompt='Create a character named John who is 23 years old.',
    functions=[create_character],
)

print(res.function_result) # { name: 'John', age: 23 }
```

### Hooks
You can hook into any part of the completion life cycle.
```python
cursive.on('completion:after', lambda result: print(
    result.data.cost.total,
    result.data.usage.total_tokens,
))

cursive.on('completion:error', lambda result: print(
    result.error,
))

cursive.ask({
    prompt: 'Can androids dream of electric sheep?',
})

# 0.0002185
# 113
```

### Embedding
You can create embeddings pretty easily with `cursive`.
```ts
embedding = cursive.embed('This should be a document.')
```
This will support different types of documents and integrations pretty soon.

### Reliability
Cursive comes with automatic retry with backoff upon failing completions, and model expanding upon exceeding context length -- which means that it tries again with a model with a bigger context length when it fails by running out of it.

You can configure this behavior by passing the `retry` and `expand` options to `useCursive`.

```python
cursive = Cursive(
    max_retries=5, # 0 disables it completely
    expand={
        'enable': True,
        'defaults_to': 'gpt-3.5-turbo-16k',
        'model_mapping': {
            'gpt-3.5-turbo': 'gpt-3.5-turbo-16k',
            'gpt-4': 'claude-2',
        },
    },
)
```

## available models
##### OpenAI
- `gpt-3.5-turbo`
- `gpt-3.5-turbo-16k`
- `gpt-4`
- `gpt-4-32k`
- Any other chat completion model version

##### Anthropic
- `claude-2`
- `claude-instant-1`
- `claude-instant-1.2`
- Any other model version

##### Anthropic
- `command`
- Any other model version (such as `command-nightly`)

## roadmap

### vendor support
- [x] Anthropic
- [x] Cohere
- [ ] Azure OpenAI models
- [ ] Huggingface 
- [ ] Replicate 
