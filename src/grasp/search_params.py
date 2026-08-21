import logging
import os
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ValidationError
from universal_ml_utils.io import dump_json, load_json

SEARCH_PARAMS_FILE = "search_params.json"


# index type specific parameters applied when searching an index; persisted next
# to the index at build time and overridable from a GRASP config, where only
# explicitly set fields override, like for knowledge graph infos
class SearchParamsBase(BaseModel):
    # subclasses declare `type` as a Literal, which doubles as the discriminator
    # in the persisted search params and in SEARCH_PARAMS_TYPES

    # fields that are persisted but describe something other than the search
    # itself; excluded from the search kwargs and never taken from configs
    non_search_fields: ClassVar[set[str]] = set()

    # parameters to pass to the index's search function; None means "leave to
    # search-rdf", so those are dropped instead of passed explicitly
    def search_kwargs(self) -> dict[str, Any]:
        return self.model_dump(exclude=self.non_search_fields, exclude_none=True)


class EmbeddingSearchParams(SearchParamsBase):
    non_search_fields: ClassVar[set[str]] = {"type", "build"}

    type: Literal["embedding"] = "embedding"
    min_score: float | None = 0.0
    rerank: float | None = 2.0
    exact: bool | None = True
    # how the params above were derived at build time; informational only,
    # cannot be set from a config
    build: dict[str, Any] | None = None


# union of all index types supporting search params; extend this together with
# SEARCH_PARAMS_TYPES below once other index types gain parameters
SearchParams = EmbeddingSearchParams

SEARCH_PARAMS_TYPES: dict[str, type[SearchParams]] = {
    "embedding": EmbeddingSearchParams,
}


# search params class for an index type, None if it has no params
def search_params_cls(index_type: str) -> type[SearchParams] | None:
    return SEARCH_PARAMS_TYPES.get(index_type)


# parameters that only matter while building an embedding index, either because
# they shape the index itself or because they determine the search params
# derived from it
class EmbeddingBuildParams(BaseModel):
    type: Literal["embedding"] = "embedding"

    # percentile used to derive min_score; None disables derivation and
    # leaves min_score at its default
    min_score_percentile: float | None = None
    min_score_margin: float = 0.0
    min_score_samples: int = 4096
    seed: int = 22


# Percentile of the random item-item similarity distribution that can be used as
# a min-score floor. The score at this percentile is the similarity that only
# (100 - p)% of random (i.e. unrelated) pairs exceed, so it acts as a noise
# floor: a genuine match must be more similar than ~75% of random pairs. The
# median (50) is roughly the average random similarity and barely filters noise.
# Values above the tail start cutting genuine matches because the random
# distribution is narrow (p50 to p90 spans only ~0.1 cosine).
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


# search params to persist for a freshly built embedding index; min_score is
# derived from the embeddings if a percentile is configured, and params set
# explicitly in `search` take precedence over the derived ones
def build_embedding_search_params(
    embeddings: np.ndarray,
    build: EmbeddingBuildParams | None = None,
    search: EmbeddingSearchParams | None = None,
) -> EmbeddingSearchParams:
    build = build or EmbeddingBuildParams()
    params = EmbeddingSearchParams()

    if build.min_score_percentile is not None:
        derived = estimate_embedding_min_score(
            embeddings,
            num_samples=build.min_score_samples,
            percentile=build.min_score_percentile,
            margin=build.min_score_margin,
            seed=build.seed,
        )
        if derived is not None:
            params.min_score = derived

    if search is not None:
        params = merge_search_params(params, search)

    params.build = build.model_dump(exclude={"type"})
    return params


# override only the fields explicitly set on `override`
def merge_search_params(base: SearchParams, override: SearchParams) -> SearchParams:
    update = override.model_dump(exclude_unset=True, exclude=override.non_search_fields)
    if not update:
        return base

    return base.model_copy(update=update)


# search params for an index of the given type, starting from the persisted
# params and applying the override from the config
def resolve_search_params(
    index_type: str,
    base: SearchParams | None = None,
    override: dict[str, Any] | None = None,
    name: str = "index",
    logger: logging.Logger | None = None,
) -> SearchParams | None:
    cls = search_params_cls(index_type)

    if cls is None:
        if override and logger is not None:
            logger.warning(
                f'Search params are not supported for "{index_type}" indices, '
                f'ignoring the params configured for "{name}"'
            )
        return None

    if base is not None and not isinstance(base, cls):
        if logger is not None:
            logger.warning(
                f'Ignoring persisted "{base.type}" search params for "{name}", '
                f'which is a "{index_type}" index'
            )
        base = None

    params = base if base is not None else cls()

    if override:
        invalid = cls.non_search_fields.intersection(override)
        if invalid:
            raise ValueError(
                f"Cannot set {sorted(invalid)} in the search params for "
                f'"{name}", they are determined at build time'
            )

        try:
            validated = cls.model_validate(override)
        except ValidationError as e:
            raise ValueError(
                f'Invalid search params for "{name}" ({index_type} index): {e}'
            ) from e

        params = merge_search_params(params, validated)

    if logger is not None:
        logger.debug(f'Search params for "{name}": {params.model_dump_json()}')

    return params


# search params for an index, from the params persisted in params_dir with the
# override from the config applied
def resolve_index_search_params(
    index_type: str,
    params_dir: str,
    override: dict[str, Any] | None = None,
    name: str = "index",
    logger: logging.Logger | None = None,
) -> SearchParams | None:
    base = None
    if search_params_cls(index_type) is not None:
        base = load_search_params(params_dir)

    return resolve_search_params(
        index_type,
        base,
        override,
        name=name,
        logger=logger,
    )


def search_params_path(index_dir: str) -> str:
    return os.path.join(index_dir, SEARCH_PARAMS_FILE)


def write_search_params(params: SearchParams, index_dir: str) -> None:
    dump_json(params.model_dump(), search_params_path(index_dir))


def load_search_params(index_dir: str) -> SearchParams | None:
    path = search_params_path(index_dir)
    if not os.path.exists(path):
        return None

    raw = dict(load_json(path))
    # indices built before the build/search param split stored the min score
    # derivation under "calibration"
    if "calibration" in raw:
        raw["build"] = raw.pop("calibration")

    typ = raw.get("type", "embedding")
    cls = search_params_cls(typ)
    if cls is None:
        raise ValueError(f"Unknown search params type: {typ}")

    return cls.model_validate(raw)
