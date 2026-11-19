from grasp.model import Message, Response, ToolCall
from grasp.tasks.sparql_qa import AnswerArgs, CancelArgs, get_answer_or_cancel


def run(*responses: Response) -> list[Message]:
    return [
        Message.system("system"),
        Message(role="user", content="question"),
        *(Message(role="assistant", content=response) for response in responses),
    ]


def call(name: str, **args) -> ToolCall:
    return ToolCall(id=name, name=name, args=args)


def test_answer_call():
    messages = run(
        Response(id="1", tool_calls=[call("cancel", explanation="no")]),
        Response(
            id="2",
            tool_calls=[call("answer", kg="wd", sparql="ASK {}", answer="yes")],
        ),
        # invalid calls are no answers
        Response(id="3", tool_calls=[call("answer", kg="wd")]),
    )
    assert get_answer_or_cancel(messages, "fb") == AnswerArgs(
        kg="wd",
        sparql="ASK {}",
        answer="yes",
    )


def test_answer_from_message():
    cancel = '{"explanation": "no", "best_attempt": null}'
    messages = run(Response(id="1", message=f"```json\n{cancel}\n```"))
    assert get_answer_or_cancel(messages, "fb") == CancelArgs(explanation="no")

    # a query without kg is for the default one
    message = "The query:\n```sparql\nASK {}\n```"
    messages = run(Response(id="1", message=message))
    assert get_answer_or_cancel(messages, "fb") == AnswerArgs(
        kg="fb",
        sparql="ASK {}",
        answer=message,
    )


def test_answer_from_execute():
    messages = run(
        Response(
            id="1",
            message="done",
            tool_calls=[call("execute", kg="wd", sparql="ASK {}")],
        ),
    )
    assert get_answer_or_cancel(messages, "fb") == AnswerArgs(
        kg="wd",
        sparql="ASK {}",
        answer="done",
    )
    assert get_answer_or_cancel(run(Response(id="1", message="hi")), "fb") is None
