def shared_rules() -> list[str]:
    return [
        "As you hit the limits on the number of notes and their length, \
gradually generalize your notes, discard unnecessary details, and move \
notes that can be useful across knowledge graphs to the general section.",
    ]


# Re-exports kept after `shared_rules` so the submodules can import it during
# their own initialization without hitting a circular-import error.
from grasp.tasks.exploration.functional import (  # noqa: E402
    FunctionalExplorationState,
    FunctionalExplorationTask,
)
from grasp.tasks.exploration.structural import (  # noqa: E402
    StructuralExplorationState,
    StructuralExplorationTask,
)
__all__ = [
    "FunctionalExplorationState",
    "FunctionalExplorationTask",
    "StructuralExplorationState",
    "StructuralExplorationTask",
    "shared_rules",
]
