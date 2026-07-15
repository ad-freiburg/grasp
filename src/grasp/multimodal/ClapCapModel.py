import torch
import numpy as np
from msclap import CLAP


class ClapCapModel:
    """Audio-Captioning mit Microsoft CLAP (clapcap).

    Erzeugt Freitextbeschreibungen für Audiodateien statt Embeddings.
    Benötigt: pip install msclap

    Args:
        version: Modellversion; ``'clapcap'`` für Captioning,
                 ``'2023'`` für Embeddings.
        use_cuda: CUDA verwenden, falls verfügbar.
    """

    def __init__(self, version: str = "clapcap", use_cuda: bool | None = None):
        if use_cuda is None:
            use_cuda = torch.cuda.is_available()

        self.model = CLAP(version=version, use_cuda=use_cuda)

    def generate_captions(self, file_paths: list[str]) -> list[str]:
        """Erzeugt Audiobeschreibungen für eine Liste von Audiodateien.

        Args:
            file_paths: Pfade zu Audiodateien (wav, mp3, flac, …).

        Returns:
            Liste von natürlichsprachigen Beschreibungen.
        """
        return self.model.generate_caption(file_paths)

    def embed_audio(self, file_paths: list[str]) -> np.ndarray:
        """Audio-Embeddings aus Dateipfaden"""
        embs = self.model.get_audio_embeddings(file_paths)
        return np.array(embs, dtype=np.float32)

    def embed_text(self, texts: list[str]) -> np.ndarray:
        embs = self.model.get_text_embeddings(texts)
        return np.array(embs, dtype=np.float32)
