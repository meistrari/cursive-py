from enum import Enum
from typing import Any, Callable, Literal, Optional

from cursive.compat.pydantic import BaseModel as PydanticBaseModel, Field
from cursive.function import CursiveFunction
from cursive.utils import random_id


class BaseModel(PydanticBaseModel):
    class Config(PydanticBaseModel.Config):
        arbitrary_types_allowed = True
        protected_namespaces = ()


class CursiveLanguageModel(Enum):
    GPT4 = "gpt4"
    GPT4_32K = "gpt4-32k"
    GPT3_5_TURBO = "gpt-3.5-turbo"
    GPT3_5_TURBO_16K = "gpt-3.5-turbo-16k"
    CLAUDE_2 = "claude-2"
    CLAUDE_INSTANT_1_2 = "claude-instant-1.2"
    CLAUDE_INSTANT_1 = "claude-instant-1.2"
    COMMAND = "command"
    COMMAND_NIGHTLY = "command-nightly"


class CursiveAskUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class CursiveAskCost(BaseModel):
    completion: float
    prompt: float
    total: float
    version: str


Role = Literal[
    "system",
    "user",
    "assistant",
    "function",
]


class ChatCompletionRequestMessageFunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class CompletionMessage(BaseModel):
    id: Optional[str] = None
    role: Role
    content: Optional[str] = None
    name: Optional[str] = None
    function_call: Optional[ChatCompletionRequestMessageFunctionCall] = None

    def __init__(self, **data):
        super().__init__(**data)
        if not self.id:
            self.id = random_id()


class CursiveSetupOptionsExpand(BaseModel):
    enabled: Optional[bool] = None
    defaults_to: Optional[str] = None
    model_mapping: Optional[dict[str, str]] = None


class CursiveErrorCode(Enum):
    function_call_error = ("function_call_error",)
    completion_error = ("completion_error",)
    invalid_request_error = ("invalid_request_error",)
    embedding_error = ("embedding_error",)
    unknown_error = ("unknown_error",)


class CursiveError(Exception):
    name = "CursiveError"

    def __init__(
        self,
        message: str,
        details: Any,
        code: CursiveErrorCode,
    ):
        self.message = message
        self.details = details
        self.code = code
        super().__init__(self.message)


class CursiveEnrichedAnswer(BaseModel):
    error: CursiveError | None = None
    usage: CursiveAskUsage | None = None
    model: str
    id: str | None = None
    choices: Optional[list[Any]] = None
    function_result: Any | None = None
    answer: str | None = None
    messages: list[CompletionMessage] | None = None
    cost: CursiveAskCost | None = None


CursiveAskOnToken = Callable[[dict[str, Any]], None]


class CursiveAskOptionsBase(BaseModel):
    model: Optional[str | CursiveLanguageModel] = None
    system_message: Optional[str] = None
    functions: Optional[list[CursiveFunction]] = None
    function_call: Optional[str | CursiveFunction] = None
    on_token: Optional[CursiveAskOnToken] = None
    max_tokens: Optional[int] = None
    stop: Optional[list[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    best_of: Optional[int] = None
    n: Optional[int] = None
    logit_bias: Optional[dict[str, float]] = None
    user: Optional[str] = None
    stream: Optional[bool] = None


class CreateChatCompletionResponse(BaseModel):
    id: str
    model: str
    choices: list[Any]
    usage: Optional[Any] = None


class CreateChatCompletionResponseExtended(CreateChatCompletionResponse):
    function_result: Optional[Any] = None
    cost: Optional[CursiveAskCost] = None
    error: Optional[CursiveError] = None


class CursiveAskModelResponse(BaseModel):
    answer: CreateChatCompletionResponseExtended
    messages: list[CompletionMessage]


class CursiveSetupOptions(BaseModel):
    max_retries: Optional[int] = None
    expand: Optional[CursiveSetupOptionsExpand] = None
    is_using_openrouter: Optional[bool] = None


class CompletionRequestFunctionCall(BaseModel):
    name: str
    inputs: dict[str, Any] = Field(default_factory=dict)


class CompletionRequestStop(BaseModel):
    messages_seen: Optional[list[CompletionMessage]] = None
    max_turns: Optional[int] = None


class CompletionFunctions(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None


class CompletionPayload(BaseModel):
    model: str
    messages: list[CompletionMessage]
    functions: Optional[list[CompletionFunctions]] = None
    function_call: Optional[CompletionRequestFunctionCall] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = None
    stop: Optional[CompletionRequestStop] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[dict[str, float]] = None
    user: Optional[str] = None
    other: Optional[dict[str, Any]] = None


CursiveHook = Literal[
    "embedding:before",
    "embedding:after",
    "embedding:error",
    "embedding:success",
    "completion:before",
    "completion:after",
    "completion:error",
    "completion:success",
    "ask:before",
    "ask:after",
    "ask:success",
    "ask:error",
]


class CursiveHookPayload:
    data: Optional[Any]
    error: Optional[CursiveError]
    duration: Optional[float]

    def __init__(
        self,
        data: Optional[Any] = None,
        error: Optional[CursiveError] = None,
        duration: Optional[float] = None,
    ):
        self.data = data
        self.error = error
        self.duration = duration
