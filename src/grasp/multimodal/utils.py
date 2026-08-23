import os
import io
import base64
import json
import tempfile
from enum import Enum
from urllib.request import Request, urlopen
from PIL import Image
import numpy as np
from grasp.configs import Modality
from grasp.utils import FunctionCallException

_IMAGE_EXTENSION_MAP = {
    "jpg": "jpeg",
    "jpe": "jpeg",
    "tif": "tiff",
}


class ModalityTypes(str, Enum):
    BASE64 = "base64"
    URL = "url"
    FILE = "file"


MAX_IMAGE_DIMENSION = 1024  # Max Image Resolution


def image_file_to_base64(path: str, max_dimension: int) -> str:
    """
    Converts a local image path into a base64 encoded image_url
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    with open(path, "rb") as file:
        image_bytes = file.read()

    extention = os.path.splitext(path)[1].lower().lstrip(".")
    extention = _IMAGE_EXTENSION_MAP.get(extention, extention)
    content_type = "image/" + extention

    return rescale_image(image_bytes, content_type, max_dimension)


def image_url_to_base64(url: str, max_dimension: int) -> str:
    """
    Downloads and converts an external image into a base64 encoded image_url
    """
    request = Request(
        url,
        headers={"User-Agent": "GRASP https://github.com/ad-freiburg/grasp"}
    )
    try:
        with urlopen(request, timeout=10) as response:
            content_type = response.headers.get("Content-Type", "image/jpeg").split(";")[0]
            image_bytes = response.read()
    except Exception as e:
        raise FunctionCallException(f"Failed to download image from {url}: \n{e}") from e

    return rescale_image(image_bytes, content_type, max_dimension)


def audio_base64_to_file(string: str, suffix: str = ".wav") -> str:
    if string.startswith("data:"):
        string = string.split(",", 1)[1]

    raw = base64.b64decode(string)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(raw)
        return f.name


def audio_url_to_base64(url: str) -> dict:
    request = Request(
        url,
        headers={"User-Agent": "GRASP https://github.com/ad-freiburg/grasp"}
    )
    try:
        with urlopen(request, timeout=10) as response:
            content_type = response.headers.get("Content-Type", "audio/wav").split(";")[0].strip()
            audio_bytes = response.read()
    except Exception as e:
        raise FunctionCallException(f"Failed to download audio from {url}: \n{e}") from e

    format = _AUDIO_FORMAT_MAP.get(content_type)
    data = base64.b64encode(audio_bytes).decode("utf-8")
    return {"type": "input_audio", "input_audio": {"data": data, "format": format}}


def audio_file_to_base64(filepath: str) -> dict:
    with open(filepath, "rb") as file:
        audio_bytes = file.read()
        data = base64.b64encode(audio_bytes).decode("utf-8")
        file_extention = filepath.rsplit(".", 1)[-1].lower()
    return {"type": "input_audio", "input_audio": {"data": data, "format": file_extention}}


def normalize_audio_base64_input(input_data: str, fallback_format: str = "wav") -> tuple[str, str]:
    if input_data.startswith("data:"):
        header, payload = input_data.split(",", 1)
        mime = header[5:].split(";", 1)[0].strip().lower()
        audio_format = _AUDIO_FORMAT_MAP.get(mime)
        if audio_format is None and "/" in mime:
            audio_format = mime.rsplit("/", 1)[-1]
        return payload, audio_format or fallback_format

    return input_data, fallback_format


def convert_base64_to_np_array(image_url: str) -> np.ndarray:
    _, b64data = image_url.split(",", 1)
    img_bytes = base64.b64decode(b64data)
    return np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))


def rescale_image(bytes: bytes, content_type: str, max_dimension: int) -> str:
    img = Image.open(io.BytesIO(bytes))
    longest = max(img.width, img.height)

    if longest <= max_dimension:
        data = base64.b64encode(bytes).decode("utf-8")
        return f"data:{content_type};base64,{data}"

    scale = max_dimension / longest
    new_size = (int(img.width * scale), int(img.height * scale))
    img = img.resize(new_size, resample=Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    format = content_type.split("/")[-1].upper()
    format = "JPEG" if format not in ("JPEG", "PNG", "WEBP") else format
    img.save(buffer, format=format, quality=85)
    image_bytes = buffer.getvalue()
    data = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{content_type};base64,{data}"


def guess_modality_type(input: str) -> ModalityTypes:
    # Guess data_type
    input_type: ModalityTypes
    if input.startswith("http"):
        input_type = ModalityTypes.URL
    elif input.startswith("data:"):
        input_type = ModalityTypes.BASE64
    else:
        input_type = ModalityTypes.FILE
    return input_type


def is_multimodal_payload(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    if "input" not in value:
        return False
    return any(key in value for key in ("image_url", "image_input", "audio_input"))


def media_reference_hint(num_images: int, num_audio: int) -> str:
    details: list[str] = []
    if num_images > 0:
        details.extend(
            f"USER_INPUT{i} (modality='image')"
            for i in range(1, num_images + 1)
        )
    if num_audio > 0:
        start = num_images + 1
        end = num_images + num_audio + 1
        details.extend(
            f"USER_INPUT{i} (modality='audio')"
            for i in range(start, end)
        )
    if not details:
        return ""
    return (
        " [info] user appended media files. "
        "If you call analyze(...), USER_INPUT<i> indices map as follows: "
        + "; ".join(details)
        + ". "
        + "Analyze ALL given USER_INPUTs before canceling the task!"
    )


def extract_user_input(input: str, user_input: list[str]) -> str:
    if input.startswith("USER_INPUT"):
        if user_input is None:
            raise FunctionCallException("No user media input available")
        try:
            index = int(input[len("USER_INPUT"):])
        except ValueError as exc:
            raise FunctionCallException(
                f"Invalid USER_INPUT reference: {input}"
            ) from exc
        if index < 1 or index > len(user_input):
            raise FunctionCallException(
                f"USER_INPUT index out of range: {index} (available: {len(user_input)})"
            )
        return user_input[index - 1]
    else:
        return input


_AUDIO_FORMAT_MAP = {
    "application/ogg": "ogg",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/ogg": "ogg",
    "audio/webm": "webm",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
}


def unwrap_json_string_payload(raw: str) -> str:
    text = (raw or "").strip()

    if text.startswith("{") and text.endswith("}"):
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                first_string = next(
                    (value for value in payload.values() if isinstance(value, str)),
                    None,
                )
                if first_string is not None:
                    return first_string
        except json.JSONDecodeError:
            pass
    return text


IMAGE_ANALYSIS_TOOL_SCHEMA = {
    "name": "emit_image_analysis",
    "description": "Return structured visual facts extracted from the image.",
    "parameters": {
        "type": "object",
        "properties": {
            "image_type": {
                "type": "string",
                "enum": [
                    "landscape_photo",
                    "portrait",
                    "traffic_camera",
                    "presentation_slide",
                    "document_scan",
                    "chart_or_plot",
                    "map",
                    "screenshot",
                    "illustration",
                    "other",
                ],
            },
            "scene_description": {
                "type": "string",
                "description": "Short factual scene summary based only on visible evidence.",
            },
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "label": {"type": "string"},
                        "category": {"type": "string"},
                        "entity": {"type": "string"},
                        "locality": {
                            "type": "object",
                            "properties": {
                                "position": {
                                    "type": "string",
                                    "enum": [
                                        "top_left",
                                        "top_center",
                                        "top_right",
                                        "center_left",
                                        "center",
                                        "center_right",
                                        "bottom_left",
                                        "bottom_center",
                                        "bottom_right",
                                    ],
                                },
                            },
                            "required": ["position"],
                            "additionalProperties": False,
                        },
                        "properties": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "value": {"type": "string"},
                                },
                                "required": ["name", "value"],
                                "additionalProperties": False,
                            },
                        },
                        "identity_hypothesis": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "entity_type": {"type": "string"},
                                "confidence": {"type": "string"},
                                "basis": {"type": "string"},
                            },
                            "required": ["name", "entity_type", "confidence", "basis"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["id", "label", "category", "entity", "locality", "properties", "identity_hypothesis"],
                    "additionalProperties": False,
                },
            },
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject_id": {"type": "string"},
                        "predicate": {"type": "string"},
                        "object_id": {"type": "string"},
                    },
                    "required": ["subject_id", "predicate", "object_id"],
                    "additionalProperties": False,
                },
            },
            "text_visible": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "entity_id": {"type": ["string", "null"]},
                        "locality": {
                            "type": "object",
                            "properties": {
                                "position": {
                                    "type": "string",
                                    "enum": [
                                        "top_left",
                                        "top_center",
                                        "top_right",
                                        "center_left",
                                        "center",
                                        "center_right",
                                        "bottom_left",
                                        "bottom_center",
                                        "bottom_right",
                                    ],
                                },
                            },
                            "required": ["position"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["text", "entity_id", "locality"],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "image_type",
            "scene_description",
            "entities",
            "relations",
            "text_visible",
        ],
        "additionalProperties": False,
    },
    "strict": True,
}


AUDIO_ANALYSIS_TOOL_SCHEMA = {
    "name": "emit_audio_analysis",
    "description": "Return structured facts extracted from the provided audio.",
    "parameters": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Brief factual summary of the audio content.",
            },
            "language": {
                "type": "string",
                "description": "Detected language in the audio.",
            },
            "key_points": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Important content points from the audio.",
            },
            "audio_quality": {
                "type": "string",
                "description": "Audio quality and overall clarity assessment.",
            },
            "notable_noises": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Noticeable background noises or artifacts.",
            },
            "identities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "entity_type": {"type": "string"},
                        "confidence": {"type": "string"},
                        "basis": {"type": "string"},
                    },
                    "required": ["name", "entity_type", "confidence", "basis"],
                    "additionalProperties": False,
                },
                "description": "Potential identifiable entities inferred from voice or explicit mentions.",
            },
        },
        "required": [
            "summary",
            "language",
            "key_points",
            "audio_quality",
            "notable_noises",
            "identities",
        ],
        "additionalProperties": False,
    },
    "strict": True,
}
