import os
from enum import Enum
import numpy as np

from grasp.configs import GraspConfig, LLMConfig
from grasp.manager import KgManager
from grasp.model.openai import OpenAICompletionsModel
from grasp.model.base import Message, Response, ResponseMessage

from grasp.multimodal.utils import (
    image_file_to_base64,
    image_url_to_base64,
    audio_url_to_base64,
    audio_base64_to_file,
    convert_base64_to_np_array
)
from search_rdf.model.embedding import (
    OpenClipModel,
    ClapCapModel,
)


class Modality(str, Enum):
    IMAGE = "image",
    AUDIO = "audio",


class ModalityTypes(str, Enum):
    BASE64 = "base64"
    URL = "url"
    FILE = "file"


def guess_modality_type(image_url: str) -> ModalityTypes:
    # Guess data_type
    input_type: ModalityTypes
    if image_url.startswith("http"):
        input_type = ModalityTypes.URL
    elif image_url.startswith("data:"):
        input_type = ModalityTypes.BASE64
    else:
        input_type = ModalityTypes.FILE
    return input_type


def load(input: str, modality: str, datatype: str) -> dict:
    if modality == Modality.IMAGE:
        if datatype == ModalityTypes.BASE64:
            return {"type": "image_url", "image_url": {"url": input}}
        elif datatype == ModalityTypes.URL:
            data = image_url_to_base64(input)
            return {"type": "image_url", "image_url": {"url": data}}
        elif datatype == ModalityTypes.FILE:
            data = image_file_to_base64(input)
            return {"type": "image_url", "image_url": {"url": data}}
    elif modality == Modality.AUDIO:
        if datatype == ModalityTypes.BASE64:
            return {"type": "input_audio", "input_audio": {"data": input, "format": "wav"}}
        elif datatype == ModalityTypes.URL:
            return audio_url_to_base64(input)
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
    print(f"[DEBUG] verified with score: {score}")
    return score if score >= THRESHOLD_IMAGE_TO_IMAGE else 0.0


def analyze_image(image_url: str, prompt: str, models: list[LLMConfig]) -> str:
    vision_configs = models

    output_messages = {}

    for vision_config in vision_configs:
        model = OpenAICompletionsModel(vision_config)

        system_prompt = (
            "Answer with only valid JSON. "
            "No reasoning. No explanation. No extra words. "
            "Use only what is directly visible in the image. "
            "Do not infer identity unless it is strongly visually supported. "
            "If uncertain, omit the item. "
            "Return exactly this schema:\n"
            "{"
            '"entities": [string], '
            '"attributes": [string], '
            '"text_visible": [string]'
            "}\n"
            "Rules:\n"
            "- entities: salient people, objects, logos, places, or clearly recognizable identities.\n"
            "- attributes: atomic, visually verifiable phrases only; one fact per phrase; keep short.\n"
            "- text_visible: exact text seen in the image, or [] if none.\n"
            "- No full sentences.\n"
            "- No duplicates.\n"
            "- Prefer 1 to 5 items per list.\n"
            "- If nothing is visible for a field, use [].\n"
            "If you cannot comply, reply exactly: I cannot determine the answer from the image."
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

        response: Response = model.call(messages, fns=[])
        if isinstance(response.message, ResponseMessage):
            message = response.message.content
        else:
            message = response.message
        output_messages[vision_config.model] = message
    return str(output_messages)


def analyze_audio(audio_url: str, model: ClapCapModel) -> str:
    caption = model.generate_captions([audio_url])
    return "AUDIO DESCRIPTION: [" + ",".join(caption) + "]"


def analyze(
    input: str,
    modality: str,
    input_type: str,
    manager: KgManager,
    models: list[LLMConfig],
    prompt: str | None = None,
) -> str:

    modality = modality.lower()

    if "image" in modality:
        if prompt is None or not prompt.strip():
            raise ValueError("prompt is required for image analysis")

        image_payload = load(input, datatype=ModalityTypes.BASE64 if "base64" in modality else ModalityTypes.URL, modality=Modality.IMAGE)
        image_url = image_payload["image_url"]["url"]

        return analyze_image(image_url, prompt, models)

    if "audio" in modality:
        if manager.clap_model is None:
            raise ValueError("clap_model is required for audio analysis")

        temp_file = None

        try:
            if input_type == "filepath":
                file_path = input
            elif input_type == "audio_url":
                audio = audio_url_to_base64(input)
                format = audio["input_audio"]["format"]
                data = audio["input_audio"]["data"]
                file_path = audio_base64_to_file(data, format)
                temp_file = file_path
            elif input_type == "base64":
                file_path = audio_base64_to_file(input)
                temp_file = file_path
            else:
                raise ValueError(f"Unsupported input_type for audio: {input_type}")

            output = manager.clap_model.generate_captions([file_path])
            return "AUDIO DESCRIPTION: [" + ",".join(output) + "]"

        finally:
            if temp_file is not None and os.path.exists(temp_file):
                os.remove(temp_file)

    raise ValueError(f"Unsupported modality for analyze(): {modality}")
