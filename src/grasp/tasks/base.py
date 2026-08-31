from abc import ABC, abstractmethod
from typing import Any

from grasp.configs import GraspConfig
from grasp.manager import KgManager
from grasp.model import Message
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
    known: set[str]

    def __init__(
        self,
        managers: list[KgManager],
        config: GraspConfig,
        known: set[str] | None = None,
    ) -> None:
        self.managers = managers
        self.config = config
        self.state: Any = None
        self.known = known or set()

    @abstractmethod
    def system_information(self) -> str: ...

    @abstractmethod
    def rules(self) -> list[str]: ...

    @abstractmethod
    def function_definitions(self) -> list[dict]: ...

    @abstractmethod
    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        example_indices: dict | None,
    ) -> str: ...

    @abstractmethod
    def done(self, fn_name: str) -> bool: ...

    @abstractmethod
    def output(self, messages: list[Message]) -> dict | None: ...

    def setup(self, input: Any) -> str:
        # default is no state, and string input
        if isinstance(input, dict):
            input = input.get("text") or input.get("query") or input.get("input", "")
        assert isinstance(input, str), f"Input for {self.name} must be a string"
        return input

    @property
    def default_input_field(self) -> str | None:
        return None

    @classmethod
    def sample_cls(cls) -> type[Sample] | None:
        return None
