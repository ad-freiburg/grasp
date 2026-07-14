from grasp.functions import (
    ExecutionResult,
    execute_sparql,
    find_manager,
    update_known_from_selections,
)
from grasp.manager import KgManager
from grasp.sparql.item import selections_from_sparql
from grasp.sparql.types import Selection
from grasp.sparql.utils import READ_TIMEOUT, REQUEST_TIMEOUT


def prepare_sparql_result(
    sparql: str,
    kg: str,
    managers: list[KgManager],
    max_rows: int,
    max_columns: int,
    known: set[str] | None = None,
    request_timeout: float | tuple[float, float] | None = REQUEST_TIMEOUT,
    read_timeout: float | None = READ_TIMEOUT,
    sparql_result_max_rows: int | None = None,
) -> tuple[ExecutionResult, list[Selection]]:
    manager, _ = find_manager(managers, kg)

    # selections are parse-derived (no execution needed), so compute them first
    # and return them even when execution fails. This way callers still see the
    # resolved items of a query whose execution timed out or whose backend was
    # unavailable.
    selections = []
    try:
        selections = selections_from_sparql(sparql, manager)
        if known is not None:
            update_known_from_selections(known, selections, manager)
    except Exception:
        pass

    try:
        result = execute_sparql(
            managers,
            kg,
            sparql,
            max_rows,
            max_columns,
            known,
            request_timeout=request_timeout,
            read_timeout=read_timeout,
            sparql_result_max_rows=sparql_result_max_rows,
        )
    except Exception as e:
        return ExecutionResult(
            sparql=sparql,
            formatted=f"Error executing SPARQL query over {kg}:\n{str(e)}",
        ), selections

    return result, selections


def format_sparql_result(
    manager: KgManager,
    result: ExecutionResult,
    selections: list[Selection],
) -> str:
    fmt = f"SPARQL query over {manager.kg}:\n```sparql\n{result.sparql}\n```"

    fmt_sel = manager.format_selections(selections)
    if fmt_sel:
        fmt += f"\n\n{fmt_sel}"

    fmt += f"\n\nExecution result:\n{result.formatted}"
    return fmt
