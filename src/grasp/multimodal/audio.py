from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from threading import RLock

import numpy as np
import torch
from torch.nn import functional as F
import soundfile as sf
import soxr


def load_audio(path: str, sample_rate: int) -> np.ndarray:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    audio, file_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if not audio.size:
        raise ValueError(f"Empty audio file: {path}")
    if file_rate != sample_rate:
        audio = soxr.resample(audio, file_rate, sample_rate)
    return np.asarray(audio, dtype=np.float32)


class HuggingFaceAudioModel:
    def __init__(self, model: str = "laion/clap-htsat-unfused", device: str | None = None):
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder = None
        self.processor = None
        self._lock = RLock()

    def _ensure_loaded(self):
        from transformers import ClapModel, ClapProcessor

        with self._lock:
            if self.encoder is None:
                processor = ClapProcessor.from_pretrained(self.model)
                encoder = ClapModel.from_pretrained(self.model).to(self.device).eval()
                self.processor, self.encoder = processor, encoder

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        return self.encoder.config.projection_dim

    @staticmethod
    def _to_numpy(features) -> np.ndarray:
        return F.normalize(features.pooler_output.float(), p=2, dim=-1).cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def embed_text(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._ensure_loaded()
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        result = []
        with self._lock:
            for start in range(0, len(texts), batch_size):
                inputs = self.processor(text=texts[start:start + batch_size], padding=True,
                                        truncation=True, return_tensors="pt").to(self.device)
                result.append(self._to_numpy(self.encoder.get_text_features(**inputs)))
        return np.concatenate(result)

    @torch.inference_mode()
    def embed_audio(self, file_paths: list[str]) -> np.ndarray:
        self._ensure_loaded()
        if not file_paths:
            return np.empty((0, self.dim), dtype=np.float32)
        result = []
        with self._lock:
            sample_rate = self.processor.feature_extractor.sampling_rate
            for path in file_paths:
                audio = load_audio(path, sample_rate)
                inputs = self.processor(audio=audio, sampling_rate=sample_rate,
                                        return_tensors="pt").to(self.device)
                result.append(self._to_numpy(self.encoder.get_audio_features(**inputs)))
        return np.concatenate(result)


class HuggingFaceAudioCaptionModel:
    def __init__(self, model: str = "MU-NLPC/whisper-tiny-audio-captioning",
                 device: str | None = None, max_new_tokens: int = 100):
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_new_tokens = max_new_tokens
        self.encoder = None
        self.processor = None
        self._lock = RLock()

    def _ensure_loaded(self):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        with self._lock:
            if self.encoder is None:
                processor = WhisperProcessor.from_pretrained(self.model, language="en", task="transcribe")
                encoder = WhisperForConditionalGeneration.from_pretrained(self.model).to(self.device).eval()
                self.processor, self.encoder = processor, encoder

    @torch.inference_mode()
    def generate_captions(self, file_paths: list[str]) -> list[str]:
        if not file_paths:
            return []
        self._ensure_loaded()
        with self._lock:
            return [self._caption_file(path) for path in file_paths]

    def _caption_file(self, path: str) -> str:
        from transformers.generation import GenerationMixin

        sample_rate = self.processor.feature_extractor.sampling_rate
        audio = load_audio(path, sample_rate)
        chunk_size = 30 * sample_rate
        tokenizer = self.processor.tokenizer
        # Decoder prompt: start token + <|en|>/<|transcribe|> + Clotho's
        # "clotho > caption: " switching Whisper into caption mode.
        prefix = [self.encoder.config.decoder_start_token_id]
        prefix.extend(token for _, token in tokenizer.get_decoder_prompt_ids(language="en", task="transcribe"))
        prefix.extend(tokenizer.encode("clotho > caption: ", add_special_tokens=False))
        decoder_ids = torch.tensor([prefix], device=self.device)

        generation_config = deepcopy(self.encoder.generation_config)
        generation_config.forced_decoder_ids = None
        generation_config.return_timestamps = False
        generation_config.max_new_tokens = self.max_new_tokens
        generation_config.do_sample = False

        captions = []
        for start in range(0, len(audio), chunk_size):
            chunk = audio[start:start + chunk_size]
            features = self.processor.feature_extractor(
                chunk, sampling_rate=sample_rate, return_tensors="pt"
            ).input_features.to(self.device)
            output = GenerationMixin.generate(
                self.encoder, input_features=features, decoder_input_ids=decoder_ids,
                generation_config=generation_config,
            )
            # Skip the prefix columns, only new caption tokens get decoded.
            caption = tokenizer.batch_decode(output[:, len(prefix):], skip_special_tokens=True)[0].strip()
            end = (start + len(chunk)) / sample_rate
            captions.append(f"[{start / sample_rate:.1f}-{end:.1f}s] {caption}")
        return "\n".join(captions)


@lru_cache(maxsize=4)
def get_audio_caption_model(model: str, device: str | None = None,
                            max_new_tokens: int = 100) -> HuggingFaceAudioCaptionModel:
    return HuggingFaceAudioCaptionModel(model, device, max_new_tokens)
