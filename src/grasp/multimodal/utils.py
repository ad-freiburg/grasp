import os
import io
import base64
import tempfile
from enum import Enum
from urllib.request import Request, urlopen
from PIL import Image
import numpy as np
from grasp.utils import FunctionCallException


class Modality(str, Enum):
    IMAGE = "image"
    AUDIO = "audio"
    TEXT = "text"


class ModalityTypes(str, Enum):
    BASE64 = "base64"
    URL = "url"
    FILE = "file"


MAX_IMAGE_BYTES = 50 * 1024  # 50 KB Images at most


def image_file_to_base64(path: str) -> str:
    """
    Converts a local image path into a base64 encoded image_url
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    with open(path, "rb") as file:
        image_bytes = file.read()

    extention = os.path.splitext(path)[1].lower()
    content_type = "image/" + extention.lstrip(".")

    if (len(image_bytes) <= MAX_IMAGE_BYTES):
        data = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{content_type};base64,{data}"
    else:
        return resize_image(image_bytes, content_type)


def image_url_to_base64(url: str) -> str:
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

    if (len(image_bytes) <= MAX_IMAGE_BYTES):
        data = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{content_type};base64,{data}"
    else:
        return resize_image(image_bytes, content_type)


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


def convert_base64_to_np_array(image_url: str) -> np.ndarray:
    _, b64data = image_url.split(",", 1)
    img_bytes = base64.b64decode(b64data)
    return np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))


def resize_image(bytes: bytes, content_type: str) -> str:
    img = Image.open(io.BytesIO(bytes))
    scale = (MAX_IMAGE_BYTES / len(bytes)) ** 0.5
    new_size = (int(img.width * scale), int(img.height * scale))
    img = img.resize(new_size, resample=Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    format = content_type.split("/")[-1].upper()
    format = "JPEG" if format not in ("JPEG", "PNG", "WEBP") else format
    img.save(buffer, format=format, quality=85)
    image_bytes = buffer.getvalue()
    data = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{content_type};base64,{data}"


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


def media_reference_hint(num_images: int, num_audio: int) -> str:
    details: list[str] = []
    if num_images > 0:
        details.extend(
            f"USER_INPUT{i} (modality='image')"
            for i in range(1, num_images + 1)
        )
    if num_audio > 0:
        start = num_images + 1
        end = num_images + num_audio
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


_AUDIO_FORMAT_MAP = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/ogg": "ogg",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
}
