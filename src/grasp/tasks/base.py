from abc import ABC, abstractmethod
from typing import Any

from grasp.configs import GraspConfig
from grasp.manager import KgManager
from grasp.model import FunctionDefinition, Message, ToolCallResult
from grasp.provenance.state import Provenance, ProvenanceState
from grasp.examples import Sample


class FeedbackTask(ABC):
    @abstractmethod
    def feedback_system_message(
        self,
        kg_notes: dict[str, list[str]],
        notes: list[str],
    ) -> str: ...

    @abstractmethod
    def feedback_instructions(self, inputs: list[str], output: dict) -> str: ...


class GraspTask(ABC):
    name: str
    provenance: Provenance

    def __init__(
        self,
        managers: list[KgManager],
        config: GraspConfig,
        provenance: ProvenanceState | None = None,
        example_indices: dict | None = None,
        kg_notes: dict[str, list[str]] | None = None,
        notes: list[str] | None = None,
    ) -> None:
        self.managers = managers
        self.config = config
        self.state: Any = None
        # continues the provenance state of a past conversation, with the
        # notes shown to the model
        self.provenance = Provenance(kg_notes or {}, notes or [], provenance)
        self.example_indices = example_indices or {}
        # set for the last step after a limit, to relax checks on stopping
        self.last_step = False

    @abstractmethod
    def system_information(self) -> str: ...

    @abstractmethod
    def rules(self) -> list[str]: ...

    @abstractmethod
    def function_definitions(self) -> list[FunctionDefinition]: ...

    @abstractmethod
    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        example_indices: dict | None,
    ) -> str | ToolCallResult: ...

    @abstractmethod
    def done(self, fn_name: str) -> bool: ...

    @abstractmethod
    def output(self, messages: list[Message]) -> dict | None: ...

    def setup(self, input: Any) -> str:
        # default is no state, and string input
        assert isinstance(input, str), f"Input for {self.name} must be a string"
        return input

    @property
    def include_common_prefixes(self) -> bool:
        return True

    @property
    def default_input_field(self) -> str | None:
        return None

    def limit_message(self) -> str:
        # shown once the step or time limit is reached, before the last step
        return (
            "The step or time limit is reached. You have one last step, "
            "finish now with your best result."
        )

    def limit_functions(self) -> list[str] | None:
        # names of the functions offered in the last step, None for all
        return None

    @classmethod
    def sample_cls(cls) -> type[Sample] | None:
        return None
