import os
import json
from typing import Any
import numpy as np

from grasp.configs import GraspConfig, LLMConfig
from grasp.manager import KgManager
from grasp.model import get_model
from grasp.model.openai import OpenAICompletionsModel
from grasp.model.base import Message, Response, ResponseMessage
from grasp.utils import FunctionCallException

from grasp.multimodal.utils import (
    audio_file_to_base64,
    guess_modality_type,
    image_file_to_base64,
    image_url_to_base64,
    audio_url_to_base64,
    audio_base64_to_file,
    convert_base64_to_np_array,
    extract_user_input,
    ModalityTypes,
    Modality,
    IMAGE_ANALYSIS_TOOL_SCHEMA,
)
from search_rdf.model.embedding import OpenClipModel


def load(input: str, modality: str, user_input: list[str]) -> dict:
    input = extract_user_input(input, user_input)
    modality_type = guess_modality_type(input)

    if modality == Modality.IMAGE:
        if modality_type == ModalityTypes.BASE64:
            return {"type": "image_url", "image_url": {"url": input}}
        elif modality_type == ModalityTypes.URL:
            data = image_url_to_base64(input)
            return {"type": "image_url", "image_url": {"url": data}}
        elif modality_type == ModalityTypes.FILE:
            data = image_file_to_base64(input)
            return {"type": "image_url", "image_url": {"url": data}}
    elif modality == Modality.AUDIO:
        if modality_type == ModalityTypes.BASE64:
            return {"type": "input_audio", "input_audio": {"data": input, "format": "wav"}}
        elif modality_type == ModalityTypes.URL:
            return audio_url_to_base64(input)
        elif modality_type == ModalityTypes.FILE:
            return audio_file_to_base64(input)
    else:
        raise ValueError(f"Could not load input of type: {modality}")
    return {}


def verify(
        model: OpenClipModel,
        input_image_url: str,
        entity_image_url: str
        ) -> float:
    """
    returns the cosine similarity for images above the threshold, else 0
    """
    THRESHOLD_IMAGE_TO_IMAGE = 0.25

    # load images
    if input_image_url.startswith("data"):  # base64 url
        input_image = convert_base64_to_np_array(input_image_url)
    elif input_image_url.startswith("http"):
        input_image = convert_base64_to_np_array(image_url_to_base64(input_image_url))
    if entity_image_url.startswith("data"):  # base64 url
        entity_image = convert_base64_to_np_array(entity_image_url)
    elif entity_image_url.startswith("http"):
        entity_image = convert_base64_to_np_array(image_url_to_base64(entity_image_url))

    if input_image is None or entity_image is None:
        raise ValueError("input could not be loaded properly for comparison")

    # embed images
    embedding_input_image = model.embed_image([input_image])
    embedding_entity_image = model.embed_image([entity_image])

    score = float(np.dot(embedding_entity_image[0], embedding_input_image[0]))
    return score if score >= THRESHOLD_IMAGE_TO_IMAGE else 0.0


def analyze_image(image_url: str, prompt: str, models: list[LLMConfig], free_text_output: bool) -> str:
    vision_configs = models

    output_messages = {}

    for vision_config in vision_configs:
        model = get_model(vision_config)

        system_prompt = (
            "You are an image analysis engine. "
            "Use only what is directly visible in the image. "
            "Do not infer identity unless it is strongly visually supported. "
            "If uncertain, omit the item. "
            "Describe what entities or objects are in the picture, and where, "
            "Describe the attributes of the objects. "
            "Give description of what the image looks like. "
            "Describe all visible text in the image. "
        )

        messages = [
            Message.system(content=system_prompt),
            Message(
                role="user",
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            ),
        ]

        if not free_text_output:
            required_tool_config = vision_config.model_copy(
                update={"tool_choice": "required"}
            )
            response: Response = model.call(
                messages,
                fns=[IMAGE_ANALYSIS_TOOL_SCHEMA],
                config=required_tool_config,
            )

            structured_payload = None
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                if tool_call.name == IMAGE_ANALYSIS_TOOL_SCHEMA["name"]:
                    structured_payload = tool_call.args

            message = structured_payload
        else:
            response = model.call(messages, fns=[])
            if isinstance(response.message, ResponseMessage):
                message = response.message.content
            if isinstance(response.message, str):
                message = response.message
            else:
                message = ""

        output_messages[vision_config.model] = message
    return json.dumps(output_messages)


def analyze_audio(audio_url: dict, model: LLMConfig) -> str:
    model = get_model(model)

    system_prompt = """You are an audio analysis engine, evaluate the following points based on the provided audio: \
1. a brief summary, \
2. the detected language, \
3. the important content/key points, \
4. the audio quality or any noticeable noises. \
\
Do not include any introductory or closing sentences!"""

    messages = [
        Message.system(content=system_prompt),
        Message(
            role="user",
            content=[
                audio_url,
            ],
        ),
    ]

    response: Response = model.call(messages, fns=[])
    if isinstance(response.message, ResponseMessage):
        message = response.message.content
    else:
        message = response.message
    return message


def caption_audio(input: str, input_type: ModalityTypes, manager: KgManager) -> str:
    if manager.clap_model is None:
        raise FunctionCallException("clap_model is required for audio analysis")

    temp_file = None

    try:
        if input_type == ModalityTypes.FILE:
            file_path = input
        elif input_type == ModalityTypes.URL:
            audio = audio_url_to_base64(input)
            format = audio["input_audio"]["format"]
            data = audio["input_audio"]["data"]
            file_path = audio_base64_to_file(data, format)
            temp_file = file_path
        elif input_type == ModalityTypes.BASE64:
            file_path = audio_base64_to_file(input)
            temp_file = file_path
        else:
            raise FunctionCallException(
                f"Unsupported input_type for audio: {input_type}"
            )

        output = manager.clap_model.generate_captions([file_path])
        return "AUDIO DESCRIPTION: [" + ",".join(output) + "]"

    finally:
        if temp_file is not None and os.path.exists(temp_file):
            os.remove(temp_file)


def analyze(
    input: str,
    modality: Modality,
    models: list[str],
    config: GraspConfig,
    manager: KgManager | None,
    prompt: str | None = None,
    user_input: list[str] | None = None,
) -> str:
    if not models:
        raise FunctionCallException("no model choice given for analysis")

    input = extract_user_input(input, user_input)

    data_type = guess_modality_type(input)

    if modality == Modality.IMAGE:
        if prompt is None or not prompt.strip():
            raise FunctionCallException("prompt is required for image analysis")

        selected_models = [
            model
            for model in config.get_vision_models
            if model.model in models
        ]
        if not selected_models:
            raise FunctionCallException(
                "No configured vision model matches the requested models"
            )

        image_payload = load(input, modality, user_input)
        image_url = image_payload["image_url"]["url"]

        return analyze_image(image_url, prompt, selected_models, config.anser_in_free_text)

    if modality == Modality.AUDIO:
        audio_models = config.get_audio_models
        if audio_models:
            audio_url = load(input, modality, user_input)
            return analyze_audio(audio_url, audio_models[0])  # only use the first audio model

        if manager is None:
            raise FunctionCallException("kg is required for audio analysis")
        return caption_audio(input, data_type, manager)

    raise FunctionCallException(f"Unsupported modality for analyze(): {modality}")
