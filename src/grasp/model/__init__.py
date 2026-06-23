from grasp.configs import ModelConfig
from grasp.model.anthropic import AnthropicModel
from grasp.model.base import Message, Model, Response, ToolCall
from grasp.model.openai import OpenAICompletionsModel, OpenAIResponsesModel


def get_model(config: ModelConfig) -> Model:
    if config.model_provider == "openai/completions":
        return OpenAICompletionsModel(config)

    elif config.model_provider == "openai/responses":
        return OpenAIResponsesModel(config)

    elif config.model_provider == "anthropic":
        return AnthropicModel(config)

    else:
        raise ValueError(f"Unknown model provider: {config.model_provider}")


__all__ = [
    "ToolCall",
    "Message",
    "Response",
    "Model",
]
