import argparse
from collections.abc import Sequence
from datetime import datetime

from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI

from research_assistant.agent import SEPARATOR, ResearchAgent
from research_assistant.config import (
    ORACLE_DSN,
    ORACLE_PROGRAM,
    VECTOR_PASSWORD,
    VECTOR_USER,
    load_env,
)
from research_assistant.db.connection import connect_to_oracle
from research_assistant.db.schema import (
    ALL_TABLES,
    CONVERSATIONAL_TABLE,
    KNOWLEDGE_BASE_TABLE,
    drop_tables,
)
from research_assistant.db.setup import setup_oracle_database
from research_assistant.memory.manager import list_threads
from research_assistant.memory.stores import build_memory_manager
from research_assistant.tools.common import register_common_tools
from research_assistant.tools.toolbox import Toolbox

EMBEDDING_MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"
EXIT_COMMANDS = {"stop", "exit", "quit"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m research_assistant",
        description="Memory aware research assistant.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop all memory tables before starting, discarding previous runs.",
    )
    return parser.parse_args(argv)


def build_agent(conn) -> ResearchAgent:
    llm_client = OpenAI()
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    memory = build_memory_manager(conn, embedding_model)
    toolbox = Toolbox(memory, llm_client, embedding_model)
    register_common_tools(toolbox, memory, KNOWLEDGE_BASE_TABLE)
    print("✅ MemoryManager and Toolbox initialized")

    return ResearchAgent(llm_client=llm_client, memory=memory, toolbox=toolbox)


def select_thread(conn) -> str:
    """Ask which conversation thread to use, listing the ones already stored."""
    threads = list_threads(conn, CONVERSATIONAL_TABLE)

    print(f"\n{SEPARATOR}")
    if threads:
        print("💬 EXISTING THREADS")
        for thread_id, count, last_ts in threads:
            last = last_ts.strftime('%Y-%m-%d %H:%M') if last_ts else "unknown"
            print(f"  • {thread_id} — {count} messages — last active {last}")
    else:
        print("💬 No existing threads")
    print(SEPARATOR)

    try:
        choice = input("Thread id to resume, a new name, or Enter for a new thread: ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = ""

    thread_id = choice or datetime.now().strftime("%Y%m%d-%H%M%S")
    known = {str(t[0]) for t in threads}
    print(f"➡️ Using thread {thread_id}" + ("" if thread_id in known else " (new)"))
    return thread_id


def run_session(agent: ResearchAgent, conn) -> None:
    """Prompt for questions until the user asks to stop."""
    thread_id = select_thread(conn)
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
            agent.answer(question, thread_id)
        # One failing question must not end the session.
        except Exception as e:  # noqa: BLE001
            print(f"\n❌ The agent failed on that question: {e}")

    print("👋 Session ended")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    load_env()
    setup_oracle_database()

    with connect_to_oracle(
        user=VECTOR_USER,
        password=VECTOR_PASSWORD,
        dsn=ORACLE_DSN,
        program=ORACLE_PROGRAM,
    ) as conn:
        print("Using user:", conn.username)

        # Must run before build_agent, which recreates the tables.
        if args.reset:
            print("🗑️ --reset: dropping all memory tables")
            drop_tables(conn, ALL_TABLES)
        else:
            print("📚 Resuming from existing memory (use --reset to start clean)")

        run_session(build_agent(conn), conn)
