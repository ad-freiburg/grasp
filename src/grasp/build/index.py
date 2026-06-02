import os
import time

from safetensors.numpy import save_file
from search_rdf import Data, EmbeddingIndex, FuzzyIndex, KeywordIndex
from search_rdf.model import SentenceTransformerModel
from universal_ml_utils.logging import get_logger
from universal_ml_utils.ops import flatten

from grasp.manager.utils import get_auto_index_type, load_data
from grasp.search_params import build_embedding_search_params, write_search_params
from grasp.utils import get_index_dir


def build_index(
    kg: str,
    index_name: str,
    index_type: str,
    embedding_model: str | None = None,
    embedding_device: str | None = None,
    embedding_batch_size: int = 256,
    embedding_dim: int | None = None,
    log_level: str | int | None = None,
    overwrite: bool = False,
) -> None:
    logger = get_logger("GRASP INDEX", log_level)

    data_dir = os.path.join(get_index_dir(kg), index_name)
    data = load_data(data_dir)

    if index_type == "auto":
        index_type = get_auto_index_type(index_name, data)
        logger.info(f'Auto-selected index type {index_type} for index "{index_name}"')

    index_dir = os.path.join(data_dir, index_type)

    if os.path.exists(index_dir) and not overwrite:
        logger.info(
            f"Index of type {index_type} already exists at {index_dir}. Skipping build."
        )
        return

    os.makedirs(index_dir, exist_ok=True)
    start = time.perf_counter()
    logger.info(f"Building {index_type} index at {index_dir} from {len(data):,} items")

    if index_type == "keyword":
        KeywordIndex.build(data, index_dir)

    elif index_type == "fuzzy":
        FuzzyIndex.build(data, index_dir)

    elif index_type == "embedding":
        assert embedding_model is not None, (
            "Embedding model must be specified for embedding index"
        )
        embedding_path = os.path.join(index_dir, "embedding.safetensors")

        embeddings = generate_embeddings(
            data,
            embedding_path,
            model_name=embedding_model,
            device=embedding_device,
            batch_size=embedding_batch_size,
            dim=embedding_dim,
        )

        EmbeddingIndex.build(data, embedding_path, index_dir)

        params = build_embedding_search_params(embeddings)
        write_search_params(params, index_dir)
        if params.calibration is None:
            logger.info(
                f"Index too small to calibrate min_score; "
                f"using default {params.min_score:.3f}"
            )
        else:
            logger.info(f"Calibrated min_score={params.min_score:.3f}")

    else:
        raise ValueError(f"Unknown index type: {index_type}")

    end = time.perf_counter()
    logger.info(f"Index build took {end - start:.2f} seconds")


def generate_embeddings(
    data: Data,
    embedding_path: str,
    model_name: str,
    device: str | None = None,
    batch_size: int = 256,
    dim: int | None = None,
):
    model = SentenceTransformerModel(model_name, device)

    texts = list(flatten(fields for _, fields in data))
    embedding = model.embed(texts, dim, batch_size, show_progress=True)

    save_file(
        {"embedding": embedding},
        filename=embedding_path,
        metadata={"model": model_name},
    )
    return embedding
