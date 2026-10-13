"""L5: Memory Aware Agent.

Script version of L5.ipynb.
"""

import argparse
import json as json_lib
from datetime import datetime

from research_assistant.config import suppress_warnings

suppress_warnings()

parser = argparse.ArgumentParser(description="Memory aware research assistant.")
parser.add_argument(
    "--reset",
    action="store_true",
    help="Drop all memory tables before starting, discarding previous runs.",
)
args = parser.parse_args()

from research_assistant.config import load_env
from research_assistant.db.setup import setup_oracle_database
from research_assistant.db.connection import connect_to_oracle

load_env()

# One-time admin setup: configures tablespace, vector memory, and VECTOR user
setup_oracle_database()

# Connect as the VECTOR user for all subsequent operations
database_connection = connect_to_oracle(
    user="VECTOR",
    password="VectorPwd_2025",
    dsn="127.0.0.1:1521/FREEPDB1",
    program="devrel.deeplearning.course_1",
)

print("Using user:", database_connection.username)

from openai import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

client = OpenAI()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-mpnet-base-v2"
)

# ==================== MEMORY STORES ====================

CONVERSATIONAL_TABLE = "CONVERSATIONAL_MEMORY"
KNOWLEDGE_BASE_TABLE = "SEMANTIC_MEMORY"
WORKFLOW_TABLE = "WORKFLOW_MEMORY"
TOOLBOX_TABLE = "TOOLBOX_MEMORY"
ENTITY_TABLE = "ENTITY_MEMORY"
SUMMARY_TABLE = "SUMMARY_MEMORY"
TOOL_LOG_TABLE = "TOOL_LOG_MEMORY"

ALL_TABLES = [
    CONVERSATIONAL_TABLE,
    KNOWLEDGE_BASE_TABLE,
    WORKFLOW_TABLE,
    TOOLBOX_TABLE,
    ENTITY_TABLE,
    SUMMARY_TABLE,
    TOOL_LOG_TABLE]

if args.reset:
    print("🗑️ --reset: dropping all memory tables")
    for table in ALL_TABLES:
        try:
            with database_connection.cursor() as cur:
                cur.execute(f"DROP TABLE {table} PURGE")
                print(f"  - {table} (dropped)")
        except Exception as e:
            if "ORA-00942" in str(e):
                print(f"  - {table} (not exists)")
            else:
                print(f"  ✗ {table}: {e}")

    database_connection.commit()
else:
    print("📚 Resuming from existing memory (use --reset to start clean)")

from research_assistant.db.schema import create_conversational_history_table, create_tool_log_table
from research_assistant.memory.manager import list_threads

CONVERSATION_HISTORY_TABLE = create_conversational_history_table(database_connection, CONVERSATIONAL_TABLE)
TOOL_LOG_HISTORY_TABLE = create_tool_log_table(database_connection, TOOL_LOG_TABLE)

from langchain_community.vectorstores.utils import DistanceStrategy
from research_assistant.memory.stores import StoreManager

store_manager = StoreManager(
    client=database_connection,
    embedding_function=embedding_model,
    table_names={
        'knowledge_base': KNOWLEDGE_BASE_TABLE,
        'workflow': WORKFLOW_TABLE,
        'toolbox': TOOLBOX_TABLE,
        'entity': ENTITY_TABLE,
        'summary': SUMMARY_TABLE,
    },
    distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    conversational_table=CONVERSATION_HISTORY_TABLE,
    tool_log_table=TOOL_LOG_HISTORY_TABLE,
)

conversation_table = store_manager.get_conversational_table()
knowledge_base_vs = store_manager.get_knowledge_base_store()
workflow_vs = store_manager.get_workflow_store()
toolbox_vs = store_manager.get_toolbox_store()
entity_vs = store_manager.get_entity_store()
summary_vs = store_manager.get_summary_store()
tool_log_table = store_manager.get_tool_log_table()

print("✅ All stores loaded via StoreManager")

from research_assistant.memory.manager import MemoryManager
from research_assistant.tools.toolbox import Toolbox
from research_assistant.tools.common import register_common_tools

memory_manager = MemoryManager(
    conn=database_connection,
    conversation_table=conversation_table,
    knowledge_base_vs=knowledge_base_vs,
    workflow_vs=workflow_vs,
    toolbox_vs=toolbox_vs,
    entity_vs=entity_vs,
    summary_vs=summary_vs,
    tool_log_table=TOOL_LOG_HISTORY_TABLE
)

toolbox = Toolbox(memory_manager, client, embedding_model)

# arxiv search, paper fetch, summary tools
common_tools = register_common_tools(toolbox, memory_manager, KNOWLEDGE_BASE_TABLE)

print("✅ MemoryManager and Toolbox initialized")

# ==================== CONTEXT ENGINEERING ====================

from research_assistant.context.usage import calculate_context_usage, monitor_context_window
from research_assistant.context.summarization import (
    summarise_context_window,
    offload_to_summary,
    summarize_conversation,
)
from research_assistant.context.entities import extract_entities
from research_assistant.context.segments import build_memory_context
from research_assistant.rendering import render_markdown

print("✅ Context management functions loaded")

AGENT_SYSTEM_PROMPT = """
# Role
You are a memory-aware agentic research assistant with access to tools.

# Context Window Structure (Partitioned Segments)
The user input is a partitioned context window. It contains a `# Question` section followed by memory segments.
Treat each segment as a distinct memory store with a specific purpose:
- `## Conversation Memory`
- `## Knowledge Base Memory`
- `## Workflow Memory`
- `## Entity Memory`
- `## Summary Memory`

# Memory Store Semantics
- Conversation Memory: Recent thread-level dialogue and instructions. Use it for continuity, user preferences, and unresolved requests.
- Knowledge Base Memory: Retrieved documents/passages. Use it to ground factual and technical claims.
- Workflow Memory: Prior execution patterns and step sequences. Use it to plan tool usage; adapt patterns, do not copy blindly.
- Entity Memory: Named people/orgs/systems and descriptors. Use it to disambiguate references and keep naming consistent.
- Summary Memory: Compressed older context represented by summary IDs. When thread-scoped summaries exist, prefer summaries for the active thread_id.

# Summary Expansion Policy
If critical detail is only present in Summary Memory or appears ambiguous, call `expand_summary(summary_id)` before relying on it.

# Operating Rules
1. Start with the provided memory segments before using tools.
2. If segments conflict, prioritize: current `# Question` > latest Conversation Memory > Knowledge Base evidence > older summaries/workflows.
3. Use only the tools provided in this turn and choose the minimum necessary tool calls.
4. If memory is insufficient, state what is missing and then use an appropriate tool.
5. For conversation compaction, use `summarize_and_store` with `thread_id` so source conversation units are marked as summarized.
"""


def execute_tool(tool_name: str, tool_args: dict, current_thread_id: str | None = None) -> str:
    """Execute a tool by looking it up in the toolbox."""

    if tool_name not in toolbox._tools_by_name:
        return f"Error: Tool '{tool_name}' not found"

    args = dict(tool_args or {})

    # Ensure conversation summarization marks source rows in the active thread.
    if tool_name == "summarize_and_store" and "thread_id" not in args and current_thread_id is not None:
        args["thread_id"] = str(current_thread_id)

    return str(toolbox._tools_by_name[tool_name](**args) or "Done")


# ==================== OPENAI CHAT FUNCTION ====================
def call_openai_chat(messages: list, tools: list = None, model: str = "gpt-5-mini"):
    """Call OpenAI Chat Completions API with tools."""
    kwargs = {"model": model, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return client.chat.completions.create(**kwargs)


# ==================== MAIN AGENT LOOP ====================
def call_agent(query: str, thread_id: str = "1", max_iterations: int = 10) -> str:
    """Agent loop with context window monitoring and summarization."""
    thread_id = str(thread_id)
    steps = []
    summaries = []  # Track created summaries

    # 1. Build context from memory
    print("\n" + "="*50)
    print("🧠 BUILDING CONTEXT...")

    # Build memory context (excluding query for now)
    memory_context = build_memory_context(memory_manager, query, thread_id)

    # 2. Check context usage - summarize if >80%
    usage = calculate_context_usage(memory_context)
    print(f"📊 Context: {usage['percent']}% ({usage['tokens']}/{usage['max']} tokens)")

    if usage['percent'] > 80:
        print("⚠️ Context >80% - offloading conversation context to summary memory...")
        memory_context, summaries = offload_to_summary(
            memory_context,
            memory_manager,
            client,
            thread_id=thread_id,
        )
        if summaries:
            print(f"🧾 Created {len(summaries)} summary reference(s): {[s['id'] for s in summaries]}")
        usage = calculate_context_usage(memory_context)
        print(f"📊 After offload: {usage['percent']}% ({usage['tokens']}/{usage['max']} tokens)")

    # Now prepend the query (always preserved, never summarized)
    context = f"# Question\n{query}\n\n{memory_context}"

    print("====CONTEXT WINDOW=====\n")
    render_markdown(context)

    # 3. Get tools
    dynamic_tools = toolbox.schemas_for(query, k=5)
    print(f"🔧 Tools: {[t['function']['name'] for t in dynamic_tools]}")

    # 4. Store user message & extract entities
    memory_manager.write_conversational_memory(query, "user", thread_id)
    try:
        for entity in extract_entities(query, client):
            memory_manager.write_entity(entity["name"], entity["type"], entity["description"])
    except Exception:
        pass

    # 5. Agent loop
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}, {"role": "user", "content": context}]
    final_answer = ""

    print("\n🤖 AGENT LOOP")
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")

        response = call_openai_chat(messages, tools=dynamic_tools)
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]})

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json_lib.loads(tc.function.arguments)
                # Format args for display (truncate long values)
                args_display = {k: (v[:50] + '...' if isinstance(v, str) and len(v) > 50 else v)
                               for k, v in tool_args.items()}
                print(f"🛠️ {tool_name}({args_display})")

                try:
                    result = execute_tool(tool_name, tool_args, current_thread_id=thread_id)
                    status = "success"
                    error_message = None
                    steps.append(f"{tool_name}({args_display}) → success")
                except Exception as e:
                    result = f"Error: {e}"
                    status = "failed"
                    error_message = str(e)
                    steps.append(f"{tool_name}({args_display}) → failed")

                # Persist full tool output to TOOL_LOG_MEMORY
                log_id = memory_manager.write_tool_log(
                    thread_id=thread_id,
                    tool_call_id=tc.id,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    result=result,
                    status=status,
                    error_message=error_message,
                    metadata={"iteration": iteration + 1},
                )

                # Next call gets only the immediate tool result (bounded for context control)
                if len(result) > 3000:
                    result_for_llm = result[:3000] + f"\n\n[Truncated for context. Full output saved in TOOL_LOG_MEMORY as log_id: {log_id}]"
                else:
                    result_for_llm = result

                result_display = result_for_llm[:200] + "..." if len(result_for_llm) > 200 else result_for_llm
                print(f"   → {result_display}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_for_llm})
        else:
            final_answer = msg.content or ""
            print(f"\n✅ DONE ({len(steps)} tool calls)")
            break
    else:
        # Max iterations reached without final answer
        print(f"\n⚠️ WARNING: Max iterations ({max_iterations}) reached without final answer")
        final_answer = "I was unable to complete the request within the allowed iterations."

    # 6. Save workflow & entities
    if steps:
        memory_manager.write_workflow(query, steps, final_answer)
    try:
        for entity in extract_entities(final_answer, client):
            memory_manager.write_entity(entity["name"], entity["type"], entity["description"])
    except Exception:
        pass
    memory_manager.write_conversational_memory(final_answer, "assistant", thread_id)

    print("\n" + "=" * 50 + "\n💬 ANSWER:")
    render_markdown(final_answer)
    print("=" * 50)
    return final_answer


EXIT_COMMANDS = {"stop", "exit", "quit"}


def select_thread() -> str:
    """Ask which conversation thread to use, listing the ones already stored."""
    threads = list_threads(database_connection, CONVERSATION_HISTORY_TABLE)

    print("\n" + "=" * 50)
    if threads:
        print("💬 EXISTING THREADS")
        for thread_id, count, last_ts in threads:
            last = last_ts.strftime('%Y-%m-%d %H:%M') if last_ts else "unknown"
            print(f"  • {thread_id} — {count} messages — last active {last}")
    else:
        print("💬 No existing threads")
    print("=" * 50)

    try:
        choice = input("Thread id to resume, a new name, or Enter for a new thread: ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = ""

    thread_id = choice or datetime.now().strftime("%Y%m%d-%H%M%S")
    known = {str(t[0]) for t in threads}
    print(f"➡️ Using thread {thread_id}" + ("" if thread_id in known else " (new)"))
    return thread_id


def run_interactive_session():
    """Prompt for questions until the user asks to stop."""
    thread_id = select_thread()
    print(f"\nAsk a question, or type {'/'.join(sorted(EXIT_COMMANDS))} to leave.")

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            break

        try:
            call_agent(question, thread_id=thread_id)
        except Exception as e:
            print(f"\n❌ The agent failed on that question: {e}")

    print("👋 Session ended")


if __name__ == "__main__":
    try:
        run_interactive_session()
    finally:
        database_connection.close()
