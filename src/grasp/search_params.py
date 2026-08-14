import os
from typing import Literal

import numpy as np
from pydantic import BaseModel
from universal_ml_utils.io import dump_json, load_json


class EmbeddingCalibration(BaseModel):
    num_samples: int
    percentile: float
    margin: float
    seed: int


class EmbeddingSearchParams(BaseModel):
    type: Literal["embedding"] = "embedding"
    min_score: float = 0.0
    rerank: float = 2.0
    exact: bool = True
    calibration: EmbeddingCalibration | None = None


SearchParams = EmbeddingSearchParams

# Percentile of the random query-query similarity distribution used as the
# default min-score floor. The score at this percentile is the similarity that
# only (100 - p)% of random (i.e. unrelated) pairs exceed, so it acts as a noise
# floor: a genuine match must be more similar than ~75% of random pairs. The
# median (50) is roughly the average random similarity and barely filters noise.
# Kept below the tail because the random distribution is narrow (p50 to p90 spans
# only ~0.1 cosine), so a higher percentile starts cutting genuine matches.
DEFAULT_MIN_SCORE_PERCENTILE = 75.0


def estimate_embedding_min_score(
    embeddings: np.ndarray,
    num_samples: int = 4096,
    percentile: float = DEFAULT_MIN_SCORE_PERCENTILE,
    margin: float = 0.0,
    seed: int = 22,
) -> float | None:
    n = len(embeddings)
    if n < 2:
        return None

    rng = np.random.default_rng(seed)
    i_idx = rng.integers(0, n, size=num_samples)
    j_idx = rng.integers(0, n, size=num_samples)
    keep = i_idx != j_idx
    i_idx, j_idx = i_idx[keep], j_idx[keep]
    if len(i_idx) == 0:
        return None

    a = embeddings[i_idx].astype(np.float32)
    b = embeddings[j_idx].astype(np.float32)
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    scores = np.sum(a * b, axis=1)

    return float(np.percentile(scores, percentile)) + margin


def build_embedding_search_params(
    embeddings: np.ndarray,
    num_samples: int = 4096,
    percentile: float = DEFAULT_MIN_SCORE_PERCENTILE,
    margin: float = 0.0,
    seed: int = 22,
    rerank: float = 2.0,
    exact: bool = True,
) -> EmbeddingSearchParams:
    estimated = estimate_embedding_min_score(
        embeddings,
        num_samples=num_samples,
        percentile=percentile,
        margin=margin,
        seed=seed,
    )
    if estimated is None:
        return EmbeddingSearchParams(rerank=rerank, exact=exact)

    return EmbeddingSearchParams(
        min_score=estimated,
        rerank=rerank,
        exact=exact,
        calibration=EmbeddingCalibration(
            num_samples=num_samples,
            percentile=percentile,
            margin=margin,
            seed=seed,
        ),
    )


def write_search_params(params: SearchParams, index_dir: str) -> None:
    dump_json(params.model_dump(), os.path.join(index_dir, "search_params.json"))


def load_search_params(index_dir: str) -> SearchParams | None:
    path = os.path.join(index_dir, "search_params.json")
    if not os.path.exists(path):
        return None

    raw = load_json(path)
    typ = raw.get("type", "embedding")
    if typ == "embedding":
        return EmbeddingSearchParams.model_validate(raw)

    raise ValueError(f"Unknown search_params type: {typ}")
