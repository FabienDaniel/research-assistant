import copy
import json
from datetime import datetime
from itertools import repeat
from types import SimpleNamespace

import pytest

from research_assistant import agent as agent_module
from research_assistant.agent import (
    MAX_ITERATIONS,
    MAX_TOOL_RESULT_CHARS,
    MODEL,
    ResearchAgent,
)
from research_assistant.tools.toolbox import Toolbox

THREAD = "thread-1"


def answer(text):
    return SimpleNamespace(content=text, tool_calls=None)


def tool_call(name, arguments="{}", call_id="call-1"):
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id=call_id, function=function)])


class FakeLLM:
    """Plays back scripted agent replies. Entity extraction calls use another model and get no entities."""

    def __init__(self, script):
        self.script = iter(script)
        self.agent_calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        if kwargs["model"] != MODEL:
            return SimpleNamespace(choices=[SimpleNamespace(message=answer("[]"))])
        self.agent_calls.append(copy.deepcopy(kwargs))
        return SimpleNamespace(choices=[SimpleNamespace(message=next(self.script))])


class FakeMemory:
    def __init__(self, conversation_rows=()):
        self.conversation_rows = list(conversation_rows)
        self.messages = []
        self.tool_logs = []
        self.workflows = []

    def read_conversational_memory(self, thread_id):
        return self.conversation_rows

    def read_knowledge_base(self, query):
        return []

    def read_workflow(self, query):
        return []

    def read_entity(self, query):
        return []

    def read_summary_context(self, query, thread_id=None):
        return []

    def read_toolbox(self, query, k):
        return []

    def toolbox_has(self, tool_name):
        return True

    def write_conversational_memory(self, content, role, thread_id):
        self.messages.append((role, content, thread_id))

    def write_tool_log(self, **log):
        self.tool_logs.append(log)
        return "log-1"

    def write_workflow(self, query, steps, final_answer):
        self.workflows.append((query, steps, final_answer))


def make_agent(script, tools=(), memory=None):
    memory = memory or FakeMemory()
    llm = FakeLLM(script)
    toolbox = Toolbox(memory, llm, embedding_function=None)
    for tool in tools:
        toolbox.register_tool(tool)
    return ResearchAgent(llm_client=llm, memory=memory, toolbox=toolbox), llm, memory


def last_tool_message(llm):
    return llm.agent_calls[-1]["messages"][-1]


def test_plain_answer_stores_both_messages_and_no_workflow():
    agent, _, memory = make_agent([answer("Hello.")])

    assert agent.answer("Hi there", THREAD) == "Hello."

    assert memory.messages == [("user", "Hi there", THREAD), ("assistant", "Hello.", THREAD)]
    assert memory.workflows == []
    assert memory.tool_logs == []


def test_tool_result_reaches_the_model_and_is_recorded():
    def lookup(topic: str) -> str:
        return f"facts about {topic}"

    agent, llm, memory = make_agent(
        [tool_call("lookup", json.dumps({"topic": "memory"})), answer("Done.")],
        tools=[lookup],
    )

    assert agent.answer("Tell me", THREAD) == "Done."

    assert last_tool_message(llm) == {"role": "tool", "tool_call_id": "call-1", "content": "facts about memory"}
    log = memory.tool_logs[0]
    assert (log["tool_name"], log["status"], log["tool_args"]) == ("lookup", "success", {"topic": "memory"})
    assert log["metadata"] == {"iteration": 1}
    query, steps, final_answer = memory.workflows[0]
    assert (query, final_answer) == ("Tell me", "Done.")
    assert steps == ["lookup({'topic': 'memory'}) → success"]


def test_unknown_tool_is_logged_as_failed_and_the_loop_continues():
    agent, llm, memory = make_agent([tool_call("missing"), answer("Recovered.")])

    assert agent.answer("Q", THREAD) == "Recovered."

    assert last_tool_message(llm)["content"] == "Error: Tool 'missing' not found"
    assert memory.tool_logs[0]["status"] == "failed"
    assert memory.workflows[0][1] == ["missing({}) → failed"]


def test_raising_tool_is_logged_as_failed_and_the_loop_continues():
    def broken() -> str:
        raise RuntimeError("boom")

    agent, llm, memory = make_agent([tool_call("broken"), answer("Recovered.")], tools=[broken])

    assert agent.answer("Q", THREAD) == "Recovered."

    assert last_tool_message(llm)["content"] == "Error: boom"
    assert memory.tool_logs[0]["status"] == "failed"
    assert memory.tool_logs[0]["error_message"] == "boom"


@pytest.mark.parametrize("raw_args", ['{"topic": ', "null", "[1]"])
def test_unusable_arguments_are_logged_as_failed_and_the_loop_continues(raw_args):
    def lookup(topic: str) -> str:
        pytest.fail("the tool must not run")

    agent, llm, memory = make_agent([tool_call("lookup", raw_args), answer("Recovered.")], tools=[lookup])

    assert agent.answer("Q", THREAD) == "Recovered."

    assert last_tool_message(llm)["content"].startswith("Error: ")
    assert memory.tool_logs[0]["status"] == "failed"
    assert memory.messages[-1] == ("assistant", "Recovered.", THREAD)


def test_long_result_is_truncated_for_the_model_but_logged_in_full():
    full_output = "x" * (MAX_TOOL_RESULT_CHARS + 1)

    def dump() -> str:
        return full_output

    agent, llm, memory = make_agent([tool_call("dump"), answer("Done.")], tools=[dump])

    agent.answer("Q", THREAD)

    sent = last_tool_message(llm)["content"]
    assert sent.startswith("x" * MAX_TOOL_RESULT_CHARS + "\n\n[Truncated")
    assert "log_id: log-1" in sent
    assert memory.tool_logs[0]["result"] == full_output


def test_max_iterations_returns_the_fallback_answer():
    def lookup() -> str:
        return "again"

    agent, llm, memory = make_agent(repeat(tool_call("lookup")), tools=[lookup])

    final_answer = agent.answer("Q", THREAD)

    assert final_answer == "I was unable to complete the request within the allowed iterations."
    assert len(llm.agent_calls) == MAX_ITERATIONS
    assert memory.messages[-1] == ("assistant", final_answer, THREAD)


def test_summarize_and_store_defaults_to_the_active_thread():
    received = []

    def summarize_and_store(text: str = "", thread_id: str | None = None) -> str:
        received.append(thread_id)
        return "stored"

    agent, _, _ = make_agent(
        [
            tool_call("summarize_and_store"),
            tool_call("summarize_and_store", json.dumps({"thread_id": "other"})),
            answer("Done."),
        ],
        tools=[summarize_and_store],
    )

    agent.answer("Q", THREAD)

    assert received == [THREAD, "other"]


def test_full_context_is_offloaded_and_the_question_is_kept(monkeypatch):
    offloaded = []

    def fake_offload(context, memory, llm_client, thread_id):
        offloaded.append(thread_id)
        return "compact context", [{"id": "summary-1"}]

    monkeypatch.setattr(agent_module, "offload_to_summary", fake_offload)
    memory = FakeMemory(conversation_rows=[("user", "x" * 900_000, datetime(2026, 1, 1))])
    agent, llm, _ = make_agent([answer("Done.")], memory=memory)

    agent.answer("Keep this question", THREAD)

    assert offloaded == [THREAD]
    assert llm.agent_calls[0]["messages"][1]["content"] == "# Question\nKeep this question\n\ncompact context"


def test_small_context_is_not_offloaded(monkeypatch):
    monkeypatch.setattr(agent_module, "offload_to_summary", lambda *a, **k: pytest.fail("must not offload"))
    agent, _, _ = make_agent([answer("Done.")])

    assert agent.answer("Q", THREAD) == "Done."
