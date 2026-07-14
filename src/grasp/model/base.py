import json
import time
from typing import Any

from pydantic import BaseModel, Field

from grasp.configs import ModelConfig


class ToolCall(BaseModel):
    id: str
    name: str
    args: dict[str, Any]
    result: str | None = None
    error: str | None = None
    # wall-clock time in seconds spent executing this function call
    elapsed: float | None = None


class Reasoning(BaseModel):
    id: str
    content: str | None = None
    summary: str | None = None
    encrypted_content: str | None = None


class ResponseMessage(BaseModel):
    id: str
    content: str


def strip_none(s: str | None) -> str | None:
    if s is None:
        return None

    s = s.strip()
    if not s:
        return None

    return s


class Response(BaseModel):
    id: str
    message: str | ResponseMessage | None = None
    reasoning: Reasoning | None = None
    tool_calls: list[ToolCall] = []
    usage: dict | None = None
    prompt_token_ids: list[int] | None = None
    token_ids: list[int] | None = None
    token_logprobs: list[float] | None = None
    token_texts: list[str] | None = None
    # wall-clock time in seconds spent generating this response
    elapsed: float | None = None
    raw: Any = Field(default=None, exclude=True)

    @property
    def is_empty(self) -> bool:
        return self.message is None and self.reasoning is None and not self.tool_calls

    @property
    def has_content(self) -> bool:
        return self.message is not None or self.has_reasoning_content

    @property
    def has_reasoning_content(self) -> bool:
        return self.reasoning_content is not None

    @property
    def reasoning_content(self) -> str | None:
        if self.reasoning is None:
            return None
        return self.reasoning.content or self.reasoning.summary

    def get_content(self) -> dict[str, str]:
        content = {}

        if self.has_reasoning_content:
            reasoning: str = self.reasoning.content or self.reasoning.summary  # type: ignore
            content["reasoning"] = reasoning

        if isinstance(self.message, str):
            content["content"] = self.message
        elif isinstance(self.message, ResponseMessage):
            content["content"] = self.message.content

        return content

    def hash(self) -> str:
        msg: dict[str, Any] = {
            "msg": self.message.content
            if isinstance(self.message, ResponseMessage)
            else self.message,
            "reasoning": self.reasoning.model_dump(exclude={"id"})
            if self.reasoning
            else None,
            "tool_calls": sorted(
                (tc.name, json.dumps(tc.args, sort_keys=True)) for tc in self.tool_calls
            ),
        }
        return json.dumps(msg, sort_keys=True)


class Message(BaseModel):
    name: str | None = Field(default=None, exclude=True)
    role: str
    content: str | Response

    @staticmethod
    def system(content: str, name: str | None = None) -> "Message":
        return Message(name=name, role="system", content=content)

    @staticmethod
    def user(content: str, name: str | None = None) -> "Message":
        return Message(name=name, role="user", content=content)

    @staticmethod
    def assistant(content: Response, name: str | None = None) -> "Message":
        return Message(name=name, role="assistant", content=content)


class Model:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def __call__(self, *args, **kwargs) -> Response:
        start = time.monotonic()
        response = self.call(*args, **kwargs)
        response.elapsed = time.monotonic() - start
        return response

    def call(
        self,
        messages: list[Message],
        fns: list[dict],
        config: ModelConfig | None = None,
    ) -> Response:
        raise NotImplementedError
