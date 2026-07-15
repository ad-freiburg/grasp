from search_rdf import EmbeddingIndex
from search_rdf.model import (
    HuggingFaceImageModel,
    OpenClipModel,
    SentenceTransformerModel,
)
from grasp.multimodal.ClapCapModel import ClapCapModel
from grasp.multimodal.functions import (
    load,
    Modality,
)
from grasp.multimodal.utils import guess_modality_type

EmbeddingModel = HuggingFaceImageModel | OpenClipModel | SentenceTransformerModel | ClapCapModel


def get_embedding_model_key(index: EmbeddingIndex) -> str:
    assert index.model is not None, "Embedding index must have model metadata"
    provider = index.provider or "sentence-transformer"
    return f"{provider}/{index.model}"


def embed_query(
    index: EmbeddingIndex,
    query: str,
    modality: Modality,
    models: dict[str, EmbeddingModel],
) -> list[float]:
    model_key = get_embedding_model_key(index)
    model = models[model_key]

    if modality == Modality.TEXT:
        if isinstance(model, SentenceTransformerModel):
            return model.embed(query)[0].tolist()
        elif isinstance(model, OpenClipModel):
            return model.embed_text(query)[0].tolist()
        elif isinstance(model, ClapCapModel):
            return model.embed_text(query)[0].tolist()
        else:
            raise ValueError(f"Unsupported embedding model type: {type(model)} for modality: {modality}")

    elif modality == Modality.IMAGE:
        input_type = guess_modality_type(query)
        image = load(query, modality, input_type)
        if isinstance(model, OpenClipModel):
            return model.embed_image(image)[0].tolist()
        elif isinstance(model, HuggingFaceImageModel):
            return model.embed_image(image)[0].tolist()
        else:
            raise ValueError(f"Unsupported embedding model type: {type(model)} for modality: {modality}")

    elif modality == Modality.AUDIO:
        input_type = guess_modality_type(query)
        audio = load(query, modality, input_type)
        if isinstance(model, ClapCapModel):
            model.embed_audio(audio)[0].tolist()
        else:
            raise ValueError(f"Unsupported embedding model type: {type(model)} for modality: {modality}")
    else:
        raise ValueError(
            f"Unsupported querytype '{modality}'"
        )
