"""L5: Memory Aware Agent.

Script version of L5.ipynb.
"""

import argparse
import json as json_lib
from datetime import datetime

parser = argparse.ArgumentParser(description="Memory aware research assistant.")
parser.add_argument(
    "--reset",
    action="store_true",
    help="Drop all memory tables before starting, discarding previous runs.",
)
args = parser.parse_args()

import oracledb

from research_assistant.config import (
    ORACLE_DSN,
    ORACLE_PROGRAM,
    VECTOR_PASSWORD,
    VECTOR_USER,
    load_env,
)
from research_assistant.db.connection import connect_to_oracle
from research_assistant.db.setup import setup_oracle_database

load_env()

# One-time admin setup: configures tablespace, vector memory, and VECTOR user
setup_oracle_database()

# Connect as the VECTOR user for all subsequent operations
database_connection = connect_to_oracle(
    user=VECTOR_USER,
    password=VECTOR_PASSWORD,
    dsn=ORACLE_DSN,
    program=ORACLE_PROGRAM,
)

print("Using user:", database_connection.username)

from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI

client = OpenAI()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-mpnet-base-v2"
)

# ==================== MEMORY STORES ====================

from research_assistant.db.schema import (
    ALL_TABLES,
    CONVERSATIONAL_TABLE,
    KNOWLEDGE_BASE_TABLE,
    drop_tables,
)

if args.reset:
    print("🗑️ --reset: dropping all memory tables")
    drop_tables(database_connection, ALL_TABLES)
else:
    print("📚 Resuming from existing memory (use --reset to start clean)")

from research_assistant.memory.manager import list_threads
from research_assistant.memory.stores import build_memory_manager
from research_assistant.tools.common import register_common_tools
from research_assistant.tools.toolbox import Toolbox

memory_manager = build_memory_manager(database_connection, embedding_model)

toolbox = Toolbox(memory_manager, client, embedding_model)

# arxiv search, paper fetch, summary tools
common_tools = register_common_tools(toolbox, memory_manager, KNOWLEDGE_BASE_TABLE)

print("✅ MemoryManager and Toolbox initialized")

# ==================== CONTEXT ENGINEERING ====================

from research_assistant.context.entities import extract_entities
from research_assistant.context.segments import build_memory_context
from research_assistant.context.summarization import offload_to_summary
from research_assistant.context.usage import calculate_context_usage
from research_assistant.prompts import load_prompt
from research_assistant.rendering import render_markdown

print("✅ Context management functions loaded")

AGENT_SYSTEM_PROMPT = load_prompt("agent_system")


def execute_tool(tool_name: str, tool_args: dict, current_thread_id: str | None = None) -> str:
    """Execute a tool by looking it up in the toolbox."""

    args = dict(tool_args or {})

    # Ensure conversation summarization marks source rows in the active thread.
    if tool_name == "summarize_and_store" and "thread_id" not in args and current_thread_id is not None:
        args["thread_id"] = str(current_thread_id)

    return toolbox.execute(tool_name, args)


# ==================== OPENAI CHAT FUNCTION ====================
def call_openai_chat(messages: list, tools: list | None = None, model: str = "gpt-5-mini"):
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
    except oracledb.Error as e:
        print(f"⚠️ Entity write failed: {e}")

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
                # Unparseable arguments are logged and displayed as the raw string the model sent.
                tool_args = tc.function.arguments
                args_display = tool_args[:50] + '...' if len(tool_args) > 50 else tool_args

                try:
                    tool_args = json_lib.loads(tool_args)
                    # Format args for display (truncate long values)
                    args_display = {k: (v[:50] + '...' if isinstance(v, str) and len(v) > 50 else v)
                                   for k, v in tool_args.items()}
                    print(f"🛠️ {tool_name}({args_display})")
                    result = execute_tool(tool_name, tool_args, current_thread_id=thread_id)
                    status = "success"
                    error_message = None
                    steps.append(f"{tool_name}({args_display}) → success")
                # Any tool failure goes back to the model as the tool result.
                except Exception as e:  # noqa: BLE001
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
    except oracledb.Error as e:
        print(f"⚠️ Entity write failed: {e}")
    memory_manager.write_conversational_memory(final_answer, "assistant", thread_id)

    print("\n" + "=" * 50 + "\n💬 ANSWER:")
    render_markdown(final_answer)
    print("=" * 50)
    return final_answer


EXIT_COMMANDS = {"stop", "exit", "quit"}


def select_thread() -> str:
    """Ask which conversation thread to use, listing the ones already stored."""
    threads = list_threads(database_connection, CONVERSATIONAL_TABLE)

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
        # One failing question must not end the session.
        except Exception as e:  # noqa: BLE001
            print(f"\n❌ The agent failed on that question: {e}")

    print("👋 Session ended")


if __name__ == "__main__":
    try:
        run_interactive_session()
    finally:
        database_connection.close()
