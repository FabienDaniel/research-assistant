import json
from dataclasses import dataclass, field

import oracledb
from openai import OpenAI

from research_assistant.context.entities import extract_entities
from research_assistant.context.segments import build_memory_context
from research_assistant.context.summarization import offload_to_summary
from research_assistant.context.usage import calculate_context_usage
from research_assistant.memory.manager import MemoryManager
from research_assistant.prompts import load_prompt
from research_assistant.rendering import render_markdown
from research_assistant.tools.toolbox import Toolbox

MODEL = "gpt-5-mini"
MAX_ITERATIONS = 10
TOOLS_PER_QUERY = 5
OFFLOAD_THRESHOLD_PERCENT = 80
MAX_TOOL_RESULT_CHARS = 3000
ARG_DISPLAY_CHARS = 50
RESULT_DISPLAY_CHARS = 200
SEPARATOR = "=" * 50


def _truncate(text: str, limit: int) -> str:
    return text[:limit] + "..." if len(text) > limit else text


def _display_args(tool_args: dict) -> dict:
    return {
        key: _truncate(value, ARG_DISPLAY_CHARS) if isinstance(value, str) else value
        for key, value in tool_args.items()
    }


def _assistant_message(message) -> dict:
    return {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in message.tool_calls
        ],
    }


@dataclass(frozen=True)
class ResearchAgent:
    """Answers questions with a tool-calling LLM loop, reading from and writing to memory."""

    llm_client: OpenAI
    memory: MemoryManager
    toolbox: Toolbox
    system_prompt: str = field(default_factory=lambda: load_prompt("agent_system"))

    def answer(self, query: str, thread_id: str) -> str:
        """Run one full agent turn and return the final answer."""
        context = self._build_context(query, thread_id)

        tools = self.toolbox.schemas_for(query, k=TOOLS_PER_QUERY)
        print(f"🔧 Tools: {[tool['function']['name'] for tool in tools]}")

        self.memory.write_conversational_memory(query, "user", thread_id)
        self._remember_entities(query)

        final_answer, steps = self._run_loop(context, tools, thread_id)

        if steps:
            self.memory.write_workflow(query, steps, final_answer)
        self._remember_entities(final_answer)
        self.memory.write_conversational_memory(final_answer, "assistant", thread_id)

        print(f"\n{SEPARATOR}\n💬 ANSWER:")
        render_markdown(final_answer)
        print(SEPARATOR)
        return final_answer

    def _build_context(self, query: str, thread_id: str) -> str:
        """Assemble the context window, offloading conversation to summaries when it is too full."""
        print(f"\n{SEPARATOR}")
        print("🧠 BUILDING CONTEXT...")

        memory_context = build_memory_context(self.memory, query, thread_id)

        usage = calculate_context_usage(memory_context)
        print(f"📊 Context: {usage['percent']}% ({usage['tokens']}/{usage['max']} tokens)")

        if usage["percent"] > OFFLOAD_THRESHOLD_PERCENT:
            print(f"⚠️ Context >{OFFLOAD_THRESHOLD_PERCENT}% - offloading conversation context to summary memory...")
            memory_context, summaries = offload_to_summary(
                memory_context,
                self.memory,
                self.llm_client,
                thread_id=thread_id,
            )
            if summaries:
                print(f"🧾 Created {len(summaries)} summary reference(s): {[s['id'] for s in summaries]}")
            usage = calculate_context_usage(memory_context)
            print(f"📊 After offload: {usage['percent']}% ({usage['tokens']}/{usage['max']} tokens)")

        # The question goes in after the usage check so offloading can never summarize it away.
        context = f"# Question\n{query}\n\n{memory_context}"

        print("====CONTEXT WINDOW=====\n")
        render_markdown(context)
        return context

    def _remember_entities(self, text: str) -> None:
        try:
            for entity in extract_entities(text, self.llm_client):
                self.memory.write_entity(entity["name"], entity["type"], entity["description"])
        except oracledb.Error as e:
            print(f"⚠️ Entity write failed: {e}")

    def _complete(self, messages: list[dict], tools: list[dict]):
        kwargs = {"model": MODEL, "messages": messages}
        if tools:
            kwargs |= {"tools": tools, "tool_choice": "auto"}
        return self.llm_client.chat.completions.create(**kwargs).choices[0].message

    def _run_loop(self, context: str, tools: list[dict], thread_id: str) -> tuple[str, list[str]]:
        """Call the LLM and its tools until it answers. Returns the answer and the tool steps taken."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": context},
        ]
        steps = []

        print("\n🤖 AGENT LOOP")
        for iteration in range(1, MAX_ITERATIONS + 1):
            print(f"\n--- Iteration {iteration} ---")
            message = self._complete(messages, tools)

            if not message.tool_calls:
                print(f"\n✅ DONE ({len(steps)} tool calls)")
                return message.content or "", steps

            messages.append(_assistant_message(message))
            for call in message.tool_calls:
                step, tool_message = self._run_tool_call(call, iteration, thread_id)
                steps.append(step)
                messages.append(tool_message)

        print(f"\n⚠️ WARNING: Max iterations ({MAX_ITERATIONS}) reached without final answer")
        return "I was unable to complete the request within the allowed iterations.", steps

    def _run_tool_call(self, call, iteration: int, thread_id: str) -> tuple[str, dict]:
        """Run and log one tool call. Returns its workflow step and the message for the LLM."""
        name = call.function.name
        # Unparseable arguments are logged and displayed as the raw string the model sent.
        tool_args = call.function.arguments
        args_display = _truncate(tool_args, ARG_DISPLAY_CHARS)

        try:
            tool_args = json.loads(tool_args)
            args_display = _display_args(tool_args)
            print(f"🛠️ {name}({args_display})")
            result = self._execute_tool(name, tool_args, thread_id)
            status = "success"
            error_message = None
        # Any tool failure goes back to the model as the tool result.
        except Exception as e:  # noqa: BLE001
            result = f"Error: {e}"
            status = "failed"
            error_message = str(e)

        log_id = self.memory.write_tool_log(
            thread_id=thread_id,
            tool_call_id=call.id,
            tool_name=name,
            tool_args=tool_args,
            result=result,
            status=status,
            error_message=error_message,
            metadata={"iteration": iteration},
        )

        result_for_llm = result
        if len(result) > MAX_TOOL_RESULT_CHARS:
            result_for_llm = (
                result[:MAX_TOOL_RESULT_CHARS]
                + f"\n\n[Truncated for context. Full output saved in TOOL_LOG_MEMORY as log_id: {log_id}]"
            )

        print(f"   → {_truncate(result_for_llm, RESULT_DISPLAY_CHARS)}")
        step = f"{name}({args_display}) → {status}"
        return step, {"role": "tool", "tool_call_id": call.id, "content": result_for_llm}

    def _execute_tool(self, name: str, tool_args: dict, thread_id: str) -> str:
        args = dict(tool_args)
        # Conversation summarization must mark source rows in the active thread.
        if name == "summarize_and_store":
            args.setdefault("thread_id", thread_id)
        return self.toolbox.execute(name, args)
