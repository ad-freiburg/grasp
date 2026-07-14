from typing import Any, Literal

from pydantic import BaseModel, Field, conlist


class KgInfo(BaseModel):
    prefixes: dict[str, str] | None = None
    description: str | None = None
    endpoint: str | None = None
    headers: dict[str, str] | None = None
    params: dict[str, str] | None = None


class ShapeConfig(BaseModel):
    max_properties_per_class: int = 30
    dense_max_properties_per_class: int = 10
    max_targets_per_property: int = 5
    min_property_share: float = 0.01
    min_target_share: float = 0.01
    sparql_result_max_rows: int = 5_000_000
    request_timeout: float | tuple[float, float] = (6.0, 30.0)
    read_timeout: float = 10.0


class KgConfig(BaseModel):
    kg: str
    entities: Literal["fuzzy", "embedding", "keyword"] | None = "fuzzy"
    properties: Literal["fuzzy", "embedding", "keyword"] | None = "embedding"
    literals: Literal["auto", "fuzzy", "embedding", "keyword"] | None = "auto"
    shapes: ShapeConfig | None = Field(default_factory=ShapeConfig)
    notes_file: str | None = None
    examples: str | None = None

    # kg info
    info: KgInfo | None = None

    # additional indices to load
    # built via search-rdf and exposed
    # via $GRASP_INDEX_DIR/{kg}/indices.yaml
    indices: list[str] = []


class ModelConfig(BaseModel):
    seed: int | None = None

    # model parameters
    model: str = "gpt-5.4-mini"
    model_provider: Literal[
        "openai/completions",
        "openai/responses",
        "anthropic",
    ] = "openai/responses"
    model_endpoint: str | None = None
    model_api_key: str | None = Field(default=None, exclude=True)
    model_timeout: float = 120.0

    # any additional inference parameters
    model_kwargs: dict[str, Any] = {}

    # important inference parameters, supported by almost
    # all providers and models
    parallel_tool_calls: bool = False
    tool_choice: Literal["auto", "required"] = "auto"
    max_completion_tokens: int = 8192  # 8k, leaves enough space for reasoning models
    num_retries: int = 2


class JudgeConfig(ModelConfig):
    # optional knowledge graph; required when judge evaluation is run with
    # --fix-formatted so that SPARQL queries can be executed and selections
    # recomputed during reformatting
    knowledge_graph: KgConfig | None = None


class GraspConfig(ModelConfig):
    # function set, notes, and knowledge graphs
    fn_set: Literal[
        "base",
        "search",
        "search_extended",
        "search_filter",
        "search_constraints",
        "all",
    ] = "search_filter"
    notes_file: str | None = None

    knowledge_graphs: list[KgConfig] = [KgConfig(kg="wikidata")]

    # for embedding indices and example indices
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"

    # optional task specific parameters
    # map[task_name, map[param_name, param_value]]
    task_kwargs: dict[str, dict[str, Any]] = {}

    # sparql query timeouts
    sparql_connection_timeout: float = 6.0
    sparql_query_timeout: float = 30.0
    sparql_read_timeout: float = 10.0
    sparql_result_max_rows: int | None = 1_000_000

    # kg function parameters
    search_k: int = 10
    # maximum page number allowed for search pagination
    search_max_pages: int = 10
    # 10 total rows, 5 top and 5 bottom
    result_max_rows: int = 10
    # same for columns
    result_max_columns: int = 10
    # 10 total results, 10 top
    list_k: int = 10
    # force that all IRIs used in a SPARQL query
    # were previously seen
    know_before_use: bool = False

    # interaction parameters
    max_steps: int = 100
    # how many times the model is allowed to repeat the same response
    # (loop) before we give up; each repetition counts as one loop
    max_loops: int = 1

    # example parameters
    num_examples: int = 3
    force_examples: str | None = None
    random_examples: bool = False

    # shape search parameters
    num_shapes: int = 3

    # enable feedback loop
    feedback: bool = False
    max_feedbacks: int = 2
    notes_only_for_feedback: bool = False

    @property
    def sparql_request_timeout(self) -> tuple[float, float]:
        return self.sparql_connection_timeout, self.sparql_query_timeout


class SpeechToTextConfig(BaseModel):
    model: str = "gpt-4o-transcribe"
    model_endpoint: str | None = None
    model_api_key: str | None = Field(default=None, exclude=True)
    model_timeout: float = 30.0
    num_retries: int = 2
    # 2MB by default
    max_audio_bytes: int = 2 * 1024 * 1024
    rate_limit: int | None = None
    rate_limit_window: int = 60
    # ISO-639-1 language hint for the STT model; None lets it auto-detect
    language: str | None = None


class ServerConfig(GraspConfig):
    port: int = 6789
    max_connections: int = 10
    max_generation_time: int = 300
    max_idle_time: int = 300
    log_outputs: str | None = None
    log_file: str | None = None
    share: str | None = None
    rate_limit: int | None = None
    rate_limit_window: int = 60
    speech_to_text: SpeechToTextConfig | None = None


class NotesConfig(GraspConfig):
    # additional parameters specific to taking notes with GRASP
    max_notes: int = 16
    max_note_length: int = 512
    num_rounds: int = 10


class NoteTakingConfig(NotesConfig):
    # optional model config for the note taking step;
    # if set, fully replaces the parent GraspConfig's model config
    # for note taking (parent fields are not inherited)
    note_taking_model: ModelConfig | None = None


class NotesFromSamplesInput(BaseModel):
    kg: str
    file: str


class NotesFromSamplesConfig(NoteTakingConfig):
    # files with task examples
    samples: conlist(NotesFromSamplesInput, min_length=1)  # type: ignore
    samples_per_round: int = 1
    samples_per_file: int | None = None
    ignore_ground_truth: bool = False
    # if True (default), run the task agent on each sample and take notes on its
    # trace; if False, give the note-taker only the samples (input + reference)
    # and let it explore the knowledge graphs itself on top of them
    run_agent: bool = True


class NotesFromOutputsConfig(NoteTakingConfig):
    # files with outputs only
    outputs: conlist(str, min_length=1)  # type: ignore
    outputs_per_round: int = 1
    outputs_per_file: int | None = None


class NotesFromExplorationConfig(NotesConfig):
    mode: Literal["functional", "structural"] = "structural"
    questions_per_round: int = 1


class NotesGenerateQuestionsConfig(NotesConfig):
    # soft per-round target, surfaced in the system prompt only;
    # generation only stops on the agent's stop call. With num_rounds
    # already on NotesConfig, this implicitly bounds the total via
    # num_rounds * questions_per_round.
    questions_per_round: int = 1
