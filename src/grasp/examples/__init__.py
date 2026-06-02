import os
import time
from typing import Any, Type

from pydantic import BaseModel
from safetensors.numpy import save_file
from search_rdf import Data, EmbeddingIndex
from search_rdf.model import SentenceTransformerModel
from universal_ml_utils.io import dump_json, dump_jsonl, load_json, load_jsonl
from universal_ml_utils.logging import get_logger
from universal_ml_utils.ops import flatten

from grasp.configs import GraspConfig
from grasp.search_params import (
    EmbeddingSearchParams,
    build_embedding_search_params,
    load_search_params,
    write_search_params,
)


class Sample(BaseModel):
    id: str | None = None

    def input(self) -> Any:
        raise NotImplementedError

    def queries(self) -> list[str]:
        raise NotImplementedError


class ExampleIndex:
    sample_cls: Type[Sample]

    def __init__(
        self,
        data: Data,
        index: EmbeddingIndex,
        model: SentenceTransformerModel,
        samples: list[Sample],
        description: str | None = None,
        search_params: EmbeddingSearchParams | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.index = index
        self.samples = samples
        self.description = description
        self.search_params = search_params or EmbeddingSearchParams()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]

    def search(
        self,
        question: str,
        k: int = 3,
        **kwargs: Any,
    ) -> list:
        embedding = self.model.embed([question])[0]
        p = self.search_params
        merged = {
            "min_score": p.min_score,
            "exact": p.exact,
            "rerank": p.rerank,
            **kwargs,
        }
        matches = self.index.search(embedding, k, **merged)
        return [self.samples[id] for id, _, _ in matches]

    @classmethod
    def load(cls, dir: str, model: SentenceTransformerModel) -> "ExampleIndex":
        data = Data.load(os.path.join(dir, "data"))
        embedding_path = os.path.join(dir, "data", "embedding.safetensors")
        index_dir = os.path.join(dir, "index")

        index = EmbeddingIndex.load(data, embedding_path, index_dir)
        assert index.model == model.model, (
            f"Embedding model mismatch: index model {index.model}, "
            f"provided model {model.model}"
        )

        samples = [
            cls.sample_cls(**sample)
            for sample in load_jsonl(os.path.join(dir, "samples.jsonl"))
        ]

        description = None
        info_path = os.path.join(dir, "info.json")
        if os.path.exists(info_path):
            description = load_json(info_path).get("description")

        search_params = load_search_params(index_dir)
        assert search_params is None or isinstance(search_params, EmbeddingSearchParams)

        return ExampleIndex(data, index, model, samples, description, search_params)

    @classmethod
    def build(
        cls,
        examples_file: str,
        output_dir: str,
        model: SentenceTransformerModel,
        batch_size: int = 256,
        overwrite: bool = False,
        log_level: str | int | None = None,
        description: str = "",
    ) -> None:
        logger = get_logger("EXAMPLE INDEX BUILD", log_level)

        samples = [cls.sample_cls(**sample) for sample in load_jsonl(examples_file)]

        if os.path.exists(output_dir) and not overwrite:
            logger.info(f"Index directory {output_dir} already exists, skipping build")
            return

        start = time.monotonic()
        logger.info(
            f"Building example index at {output_dir} from {len(samples):,} samples"
        )
        data_dir = os.path.join(output_dir, "data")
        index_dir = os.path.join(output_dir, "index")

        samples_file = os.path.join(output_dir, "samples.jsonl")
        dump_jsonl((sample.model_dump() for sample in samples), samples_file)

        items = []
        for i, sample in enumerate(samples):
            identifier = f"sample-{i}"
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

        dump_json({"description": description}, os.path.join(output_dir, "info.json"))

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
        logger.info(f"Example index built in {end - start:.2f} seconds")


def task_to_index(task: str) -> Type[ExampleIndex]:
    if task == "sparql-qa" or task == "general-qa":
        from grasp.tasks.sparql_qa.examples import SparqlQaExampleIndex

        return SparqlQaExampleIndex

    else:
        raise ValueError(f"Unknown task {task}")


def load_example_indices(
    task: str,
    config: GraspConfig,
    model: SentenceTransformerModel | str | None = None,
) -> dict[str, ExampleIndex]:
    try:
        index_cls = task_to_index(task)
    except ValueError:
        # unsupported task
        return {}

    if isinstance(model, str):
        model = SentenceTransformerModel(model)

    indices = {}
    for kg in config.knowledge_graphs:
        if kg.examples is None:
            continue

        assert model is not None, "Model must be provided to load example indices"

        indices[kg.kg] = index_cls.load(kg.examples, model)

    return indices
