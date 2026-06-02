import os
import time
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field
from safetensors.numpy import save_file
from search_rdf import Data, EmbeddingIndex
from search_rdf.model import SentenceTransformerModel
from universal_ml_utils.io import (
    dump_json,
    dump_jsonl,
    load_json,
    load_jsonl,
    load_text,
)
from universal_ml_utils.logging import get_logger
from universal_ml_utils.ops import flatten

from grasp.search_params import (
    EmbeddingSearchParams,
    build_embedding_search_params,
    load_search_params,
    write_search_params,
)


class TargetClass(BaseModel):
    type: Literal["class"] = "class"
    iri: str
    short_iri: str
    variants: list[str] = Field(default_factory=list)
    triple_count: int = 0


class TargetLiteral(BaseModel):
    type: Literal["literal"] = "literal"
    datatype: str
    triple_count: int = 0


class TargetIri(BaseModel):
    type: Literal["iri"] = "iri"
    triple_count: int = 0


Target = TargetClass | TargetLiteral | TargetIri


class PropertyProfile(BaseModel):
    iri: str
    short_iri: str
    triple_count: int = 0
    entity_count: int = 0
    variants: list[str] = Field(default_factory=list)
    targets: list[Target] = Field(default_factory=list)


class ClassProfile(BaseModel):
    iri: str
    short_iri: str
    total_entities: int = 0
    properties: list[PropertyProfile] = Field(default_factory=list)


class ShapeSample(BaseModel):
    iri: str
    short_iri: str
    profile: ClassProfile
    label: str | None = None
    aliases: list[str] = Field(default_factory=list)

    def queries(self) -> list[str]:
        from grasp.utils import ordered_unique

        candidates = []
        if self.label:
            candidates.append(self.label)
        candidates.extend(self.aliases)
        candidates.append(self.short_iri)
        return ordered_unique(candidates)


class ShapeIndex:
    def __init__(
        self,
        data: Data,
        index: EmbeddingIndex,
        model: SentenceTransformerModel,
        samples: list[ShapeSample],
        total_classes: int | None = None,
        search_params: EmbeddingSearchParams | None = None,
    ) -> None:
        self.data = data
        self.index = index
        self.model = model
        self.samples = samples
        self.total_classes = total_classes
        self.indexed_classes = len(samples)
        self.search_params = search_params or EmbeddingSearchParams()
        self.iri_map: dict[str, ShapeSample] = {s.iri: s for s in samples}

    def search(self, query: str, k: int = 10) -> list[ShapeSample]:
        embedding = self.model.embed([query])[0]
        p = self.search_params
        matches = self.index.search(
            embedding, k, min_score=p.min_score, exact=p.exact, rerank=p.rerank
        )
        return [self.samples[id] for id, _, _ in matches]

    def get_by_iri(self, iri: str) -> ShapeSample | None:
        return self.iri_map.get(iri)

    @classmethod
    def load(cls, dir: str, model: SentenceTransformerModel) -> "ShapeIndex":
        data = Data.load(os.path.join(dir, "data"))
        embedding_path = os.path.join(dir, "data", "embedding.safetensors")
        index_dir = os.path.join(dir, "index")

        index = EmbeddingIndex.load(data, embedding_path, index_dir)
        assert index.model == model.model, (
            f"Embedding model mismatch: index model {index.model}, "
            f"provided model {model.model}"
        )

        samples = [
            ShapeSample.model_validate(s)
            for s in load_jsonl(os.path.join(dir, "samples.jsonl"))
        ]

        total_classes: int | None = None
        info_path = os.path.join(dir, "info.json")
        if os.path.exists(info_path):
            total_classes = load_json(info_path).get("total_classes")

        search_params = load_search_params(index_dir)
        assert search_params is None or isinstance(search_params, EmbeddingSearchParams)

        return cls(
            data,
            index,
            model,
            samples,
            total_classes=total_classes,
            search_params=search_params,
        )

    @classmethod
    def build(
        cls,
        samples: list[ShapeSample],
        output_dir: str,
        model: SentenceTransformerModel,
        batch_size: int = 256,
        overwrite: bool = False,
        log_level: str | int | None = None,
        total_classes: int | None = None,
    ) -> None:
        logger = get_logger("SHAPE INDEX BUILD", log_level)

        if os.path.exists(output_dir) and not overwrite:
            logger.info(f"Index directory {output_dir} already exists, skipping build")
            return

        start = time.monotonic()
        logger.info(
            f"Building shape index at {output_dir} from {len(samples):,} samples"
        )

        data_dir = os.path.join(output_dir, "data")
        index_dir = os.path.join(output_dir, "index")

        samples_file = os.path.join(output_dir, "samples.jsonl")
        dump_jsonl((s.model_dump() for s in samples), samples_file)

        items = []
        for i, sample in enumerate(samples):
            identifier = f"shape-{i}"
            fields = [{"type": "text", "value": q} for q in sample.queries()]
            items.append({"identifier": identifier, "fields": fields})

        Data.build_from_items(items, data_dir)
        data = Data.load(data_dir)

        texts = list(flatten(fields for _, fields in data))
        embedding = model.embed(texts, batch_size=batch_size, show_progress=True)

        embedding_path = os.path.join(data_dir, "embedding.safetensors")
        save_file(
            {"embedding": embedding},
            filename=embedding_path,
            metadata={"model": model.model},
        )

        EmbeddingIndex.build(data, embedding_path, index_dir)

        info: dict = {"indexed_classes": len(samples)}
        if total_classes is not None:
            info["total_classes"] = total_classes
        dump_json(info, os.path.join(output_dir, "info.json"))

        params = build_embedding_search_params(embedding)
        write_search_params(params, index_dir)
        if params.calibration is None:
            logger.info(
                f"Index too small to calibrate min_score; "
                f"using default {params.min_score:.3f}"
            )
        else:
            logger.info(f"Calibrated min_score={params.min_score:.3f}")

        end = time.monotonic()
        logger.info(f"Shape index built in {end - start:.2f} seconds")


@dataclass
class Shapes:
    pattern: str | None = None
    index: ShapeIndex | None = None
    description: str | None = None
    total_classes: int | None = None


def load_setup_description(shapes_dir: str) -> str | None:
    setup_path = os.path.join(shapes_dir, "setup.json")
    if not os.path.exists(setup_path):
        return None
    return load_json(setup_path).get("description")


def load_shapes(shapes_dir: str, model: SentenceTransformerModel) -> Shapes | None:
    pattern = None
    pattern_file = os.path.join(shapes_dir, "pattern.sparql")
    if os.path.exists(pattern_file):
        pattern = load_text(pattern_file)

    index = None
    index_dir = os.path.join(shapes_dir, "index")
    if os.path.exists(index_dir):
        index = ShapeIndex.load(index_dir, model)

    if pattern is None and index is None:
        return None

    return Shapes(
        pattern=pattern,
        index=index,
        description=load_setup_description(shapes_dir),
        total_classes=index.total_classes if index is not None else None,
    )
