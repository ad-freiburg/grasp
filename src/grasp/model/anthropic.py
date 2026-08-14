from typing import Any
from uuid import uuid4

from anthropic import Anthropic
from anthropic.types import Message as AnthropicMessage

from grasp.configs import ModelConfig
from grasp.model.base import (
    Message,
    Model,
    Reasoning,
    Response,
    ResponseMessage,
    ToolCall,
    check_api_response,
)
from grasp.model.openai import coerce_tool_call_args


class AnthropicModel(Model):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.client = Anthropic(
            base_url=config.model_endpoint,
            api_key=config.model_api_key,
            timeout=config.model_timeout,
            max_retries=config.num_retries,
        )

    @staticmethod
    def prepare_messages(
        messages: list[Message],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        # Anthropic keeps the system prompt separate from the message list and
        # expects alternating user/assistant turns, with tool results delivered
        # as tool_result content blocks inside a following user message.
        system_parts: list[str] = []
        msgs: list[dict[str, Any]] = []

        for msg in messages:
            if isinstance(msg.content, str):
                if msg.role == "system":
                    system_parts.append(msg.content)
                else:
                    # user input and feedback
                    msgs.append({"role": "user", "content": msg.content})
                continue

            assert isinstance(msg.content, Response)
            cnt = msg.content

            if cnt.raw is not None:
                # round-trip the original content blocks verbatim so thinking
                # signatures are preserved (required by the API when thinking
                # is enabled and tool calls are present)
                assert isinstance(cnt.raw, AnthropicMessage)
                content: Any = cnt.raw.content
            else:
                # rebuild assistant content from non-raw response, e.g. when the
                # history is provided through the server API. Thinking blocks are
                # omitted here since their signatures are not retained.
                blocks: list[dict[str, Any]] = []
                text = (
                    cnt.message.content
                    if isinstance(cnt.message, ResponseMessage)
                    else cnt.message
                )
                if text:
                    blocks.append({"type": "text", "text": text})
                for tool_call in cnt.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tool_call.id,
                            "name": tool_call.name,
                            "input": tool_call.args,
                        }
                    )
                content = blocks

            msgs.append({"role": "assistant", "content": content})

            # collect all tool results into a single following user message
            tool_results = []
            for tool_call in cnt.tool_calls:
                assert tool_call.result is not None, "Expected tool call result"
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": tool_call.result,
                    }
                )
            if tool_results:
                msgs.append({"role": "user", "content": tool_results})

        system = "\n\n".join(system_parts) if system_parts else None
        return system, msgs

    def call(
        self,
        messages: list[Message],
        fns: list[dict],
        config: ModelConfig | None = None,
    ) -> Response:
        if config is None:
            config = self.config

        system, msgs = self.prepare_messages(messages)

        tools = [
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn["parameters"],
            }
            for fn in fns
        ]

        kwargs: dict[str, Any] = {
            "model": config.model,
            "max_tokens": config.max_completion_tokens,
            "messages": msgs,
            **config.model_kwargs,
        }

        # automatic prompt caching: Anthropic places the cache breakpoint on the
        # last cacheable block and moves it forward as the conversation grows, so
        # every step of the agentic loop reads the prior turns from cache and only
        # writes the new delta.
        # https://platform.claude.com/docs/en/build-with-claude/prompt-caching
        kwargs.setdefault("cache_control", {"type": "ephemeral"})

        # an explicit breakpoint on the system block additionally pins the large,
        # stable system+tools prefix (tools render before system) as its own cache
        # anchor, so it is reused across separate runs over the same knowledge
        # graph, not only within a single conversation.
        if system is not None:
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = {
                "type": "auto" if config.tool_choice == "auto" else "any",
                "disable_parallel_tool_use": not config.parallel_tool_calls,
            }

        response = self.client.messages.create(**kwargs)

        check_api_response(response, AnthropicMessage, config.model_endpoint)

        return self.convert(response, fns)

    @staticmethod
    def convert(response: AnthropicMessage, fns: list[dict]) -> Response:
        message = None
        reasoning = None
        tool_calls = []
        fn_schemas = {fn["name"]: fn for fn in fns}

        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)

            elif block.type == "thinking":
                reasoning = Reasoning(id=uuid4().hex, content=block.thinking or None)

            elif block.type == "redacted_thinking":
                # keep a placeholder so has_reasoning stays consistent; the
                # actual content is preserved via the raw round-trip
                reasoning = Reasoning(id=uuid4().hex)

            elif block.type == "tool_use":
                args = block.input if isinstance(block.input, dict) else {}
                schema = fn_schemas.get(block.name)
                if schema is not None:
                    args = coerce_tool_call_args(args, schema)
                tool_calls.append(ToolCall(id=block.id, name=block.name, args=args))

            else:
                raise ValueError(f"Unexpected content block type: {block.type}")

        if text_parts:
            message = ResponseMessage(id=response.id, content="\n\n".join(text_parts))

        return Response(
            id=response.id,
            message=message,
            reasoning=reasoning,
            tool_calls=tool_calls,
            usage=response.usage.model_dump(),
            raw=response,
        )
