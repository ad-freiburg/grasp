import json
from typing import Any
from uuid import uuid4

from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.responses import Response as OpenAIResponse

from grasp.configs import ModelConfig
from grasp.model.base import (
    Message,
    Model,
    Reasoning,
    Response,
    ResponseMessage,
    ToolCall,
    strip_none,
)


def coerce_nullable_strings(value: Any, spec: dict) -> Any:
    # vLLM models without strict mode sometimes emit the string "None"/"null"
    # instead of JSON null for nullable string parameters; coerce them back
    # based on the declared schema, recursing into nested objects and arrays
    typ = spec.get("type")

    if (
        isinstance(typ, list)
        and "string" in typ
        and "null" in typ
        and isinstance(value, str)
        and value.lower() in ("none", "null")
    ):
        return None

    if isinstance(value, dict) and (
        typ == "object" or (isinstance(typ, list) and "object" in typ)
    ):
        props = spec.get("properties", {})
        return {
            k: coerce_nullable_strings(v, props[k]) if k in props else v
            for k, v in value.items()
        }

    if isinstance(value, list) and (
        typ == "array" or (isinstance(typ, list) and "array" in typ)
    ):
        items_spec = spec.get("items")
        if isinstance(items_spec, dict):
            return [coerce_nullable_strings(v, items_spec) for v in value]

    return value


def coerce_tool_call_args(args: dict, fn_schema: dict) -> dict:
    props = fn_schema.get("parameters", {}).get("properties", {})
    return {
        k: coerce_nullable_strings(v, props[k]) if k in props else v
        for k, v in args.items()
    }


class OpenAICompletionsModel(Model):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.client = OpenAI(
            base_url=config.model_endpoint,
            api_key=config.model_api_key,
            timeout=config.model_timeout,
            max_retries=config.num_retries,
        )

    @staticmethod
    def prepare_messages(messages: list[Message]) -> list[dict[str, Any]]:
        msgs = []
        for msg in messages:
            if isinstance(msg.content, str):
                msgs.append(msg.model_dump())
                continue

            if msg.content.raw is not None:
                assert isinstance(msg.content.raw, ChatCompletion)
                cmpl_msg = msg.content.raw.choices[0].message
            else:
                # rebuild message content from non-raw content
                # for example, when given via the api in the server
                assert isinstance(msg.content, Response)
                cnt = msg.content
                cmpl_msg = {
                    "id": cnt.id,
                    "role": msg.role,
                    "content": cnt.message.content
                    if isinstance(cnt.message, ResponseMessage)
                    else cnt.message,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": json.dumps(tool_call.args),
                            },
                        }
                        for tool_call in cnt.tool_calls
                    ],
                }

            msgs.append(cmpl_msg)

            for tool_call in msg.content.tool_calls:
                assert tool_call.result is not None, "Expected tool call result"
                msgs.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "content": tool_call.result,
                    }
                )

        return msgs

    def call(
        self,
        messages: list[Message],
        fns: list[dict],
        config: ModelConfig | None = None,
    ) -> Response:
        if config is None:
            config = self.config

        response: ChatCompletion = self.client.chat.completions.create(
            model=config.model,
            messages=self.prepare_messages(messages),  # type: ignore
            tools=[{"type": "function", "function": fn} for fn in fns],  # type: ignore
            tool_choice=config.tool_choice,  # type: ignore
            parallel_tool_calls=config.parallel_tool_calls,
            max_completion_tokens=config.max_completion_tokens,
            **config.model_kwargs,
        )

        # convert completions api response to our response
        if not response.choices:
            return Response(id=response.id, message=None, tool_calls=[], raw=response)

        choice = response.choices[0]  # type: ignore
        if choice.finish_reason not in ["tool_calls", "stop", "length"]:
            raise ValueError(f"Unexpected finish reason {choice.finish_reason}")

        message = strip_none(choice.message.content)
        reasoning = None
        reasoning_content = getattr(
            choice.message, "reasoning_content", None
        ) or getattr(choice.message, "reasoning", None)
        if reasoning_content is not None:
            reasoning = Reasoning(
                id=uuid4().hex,
                content=strip_none(reasoning_content),
            )

        fn_schemas = {fn["name"]: fn for fn in fns}
        tool_calls = []
        for tool_call in choice.message.tool_calls or []:
            if tool_call.type != "function":
                continue

            assert tool_call.function.name is not None

            args = json.loads(tool_call.function.arguments)
            schema = fn_schemas.get(tool_call.function.name)
            if schema is not None:
                args = coerce_tool_call_args(args, schema)

            tool_calls.append(
                ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    args=args,
                )
            )

        # Extract logprobs from choice.logprobs.content (standard OpenAI format)
        token_logprobs = None
        token_texts = None
        logprobs = getattr(choice, "logprobs", None)
        if logprobs and getattr(logprobs, "content", None):
            token_logprobs = [entry.logprob for entry in logprobs.content]
            token_texts = [entry.token for entry in logprobs.content]

        # Extract token_ids and prompt_token_ids from provider_specific_fields (vLLM extension)
        token_ids = None
        prompt_token_ids = None
        if (
            hasattr(choice, "provider_specific_fields")
            and choice.provider_specific_fields  # type: ignore
        ):
            token_ids = choice.provider_specific_fields.get("token_ids")  # type: ignore
            prompt_token_ids = choice.provider_specific_fields.get("prompt_token_ids")  # type: ignore
        # Also check response-level attributes
        if token_ids is None and hasattr(response, "token_ids"):
            token_ids = response.token_ids  # type: ignore
        if prompt_token_ids is None and hasattr(response, "prompt_token_ids"):
            prompt_token_ids = response.prompt_token_ids  # type: ignore

        return Response(
            id=response.id,
            message=message,
            reasoning=reasoning,
            tool_calls=tool_calls,
            usage=response.usage.model_dump(),  # type: ignore
            prompt_token_ids=prompt_token_ids,
            token_ids=token_ids,
            token_logprobs=token_logprobs,
            token_texts=token_texts,
            raw=response,
        )


class OpenAIResponsesModel(Model):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.client = OpenAI(
            base_url=config.model_endpoint,
            api_key=config.model_api_key,
            timeout=config.model_timeout,
            max_retries=config.num_retries,
        )

    @staticmethod
    def prepare_input(messages: list[Message]) -> list[dict[str, Any]]:
        msgs = []

        for msg in messages:
            if isinstance(msg.content, str):
                msgs.append(msg.model_dump())
                continue

            if msg.content.raw is not None:
                assert isinstance(msg.content.raw, OpenAIResponse), (
                    f"Unexpected message:\n{msg.model_dump_json(indent=2)}"
                )
                msgs.extend(msg.content.raw.output)
            else:
                # rebuild message content from non-raw content
                # for example, when given via the api in the server
                assert isinstance(msg.content, Response)
                cnt = msg.content

                if isinstance(cnt.message, str):
                    msgs.append(
                        {
                            "id": f"msg_{uuid4().hex}",
                            "type": "message",
                            "role": msg.role,
                            "content": [{"type": "output_text", "text": cnt.message}],
                        }
                    )

                elif isinstance(cnt.message, ResponseMessage):
                    msgs.append(
                        {
                            "id": cnt.message.id,
                            "type": "message",
                            "role": msg.role,
                            "content": [
                                {"type": "output_text", "text": cnt.message.content}
                            ],
                        }
                    )

                if cnt.reasoning:
                    rs = cnt.reasoning
                    rs_msg: dict = {"id": rs.id}

                    if rs.summary:
                        rs_msg["summary"] = [
                            {"type": "summary_text", "text": rs.summary}
                        ]

                    if rs.content:
                        rs_msg["content"] = [
                            {"type": "reasoning_text", "text": rs.content}
                        ]

                    if rs.encrypted_content:
                        rs_msg["encrypted_content"] = rs.encrypted_content

                    msgs.append(rs_msg)

                for tool_call in cnt.tool_calls:
                    msgs.append(
                        {
                            "call_id": tool_call.id,
                            "type": "function_call",
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.args),
                        }
                    )

            for tool_call in msg.content.tool_calls:
                assert tool_call.result is not None, "Expected tool call result"
                msgs.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call.id,
                        "output": tool_call.result,
                    }
                )

        return msgs

    def call(
        self,
        messages: list[Message],
        fns: list[dict],
        config: ModelConfig | None = None,
    ) -> Response:
        if config is None:
            config = self.config

        # use responses API
        response = self.client.responses.create(
            model=config.model,
            input=self.prepare_input(messages),  # type: ignore
            tools=[{"type": "function", **fn} for fn in fns],  # type: ignore
            tool_choice=config.tool_choice,  # type: ignore
            parallel_tool_calls=config.parallel_tool_calls,
            max_output_tokens=config.max_completion_tokens,
            **config.model_kwargs,
            store=False,
            include=["reasoning.encrypted_content", "message.input_image.image_url"],
        )

        message = None
        reasoning = None
        tool_calls = []

        for output in response.output:
            if output.type == "message" and output.content:
                message = ResponseMessage(
                    id=output.id,
                    content="\n\n".join(entry.text for entry in output.content),
                )

            elif output.type == "reasoning":
                # reasoning
                reasoning = Reasoning(id=output.id)
                if output.summary:
                    reasoning.summary = "\n\n".join(
                        entry.text for entry in output.summary
                    )

                if output.content:
                    reasoning.content = "\n\n".join(
                        entry.text for entry in output.content
                    )

                if output.encrypted_content is not None:
                    reasoning.encrypted_content = output.encrypted_content

            elif output.type == "function_call":
                # tool call
                args = json.loads(output.arguments)
                schema = next((fn for fn in fns if fn["name"] == output.name), None)
                if schema is not None:
                    args = coerce_tool_call_args(args, schema)

                tool_calls.append(
                    ToolCall(
                        id=output.call_id,
                        name=output.name,
                        args=args,
                    )
                )

            else:
                raise ValueError(f"Unknown output type {type(output)}")

        return Response(
            id=response.id,
            message=message,
            reasoning=reasoning,
            tool_calls=tool_calls,
            usage=response.usage.model_dump(),  # type: ignore
            raw=response,
        )
