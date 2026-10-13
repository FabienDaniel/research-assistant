# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A memory-aware agentic research assistant (derived from a DeepLearning.AI course, "L5"). It answers research questions over multiple sessions, persisting conversation, knowledge, workflows, entities and summaries in Oracle 23ai vector tables so later sessions reuse earlier findings.

## Commands

```bash
docker compose up -d              # Oracle Free 23 (container: oracle-free, port 1521, service FREEPDB1)
docker logs -f oracle-free        # wait for "DATABASE IS READY TO USE!"
uv run python L5.py               # interactive session, resumes existing memory
uv run python L5.py --reset       # drop all memory tables first
```

There are no tests, linter or build step. `main.py` is an unused uv stub. `L5.ipynb` is the original notebook and still imports the removed `helper.py`, so it no longer runs as-is. `L5.py` is the maintained script version.

`.env` needs `OPENAI_API_KEY` and `ORACLE_PASSWORD` (read by docker-compose). `research_assistant.config.load_env()` loads it via python-dotenv.

## Architecture

- `L5.py`: wiring and the agent loop. Connects to Oracle, builds stores, registers tools, loads `AGENT_SYSTEM_PROMPT` from `prompts/agent_system.md`, `call_agent()` and the interactive thread-selection REPL.
- `research_assistant/`: reusable machinery, one topic per module.
  - `config.py`: `.env` loading, warning/log suppression. Import it first because its module-level warnings filter must apply before torch/HuggingFace load.
  - `db/`: `setup.py` (admin bootstrap, VECTOR user, ASSM tablespace), `connection.py` (connect with retries), `schema.py` (SQL table and vector index creation), `diagnostics.py` (vector object listing and cleanup).
  - `memory/`: `manager.py` (`MemoryManager`, `list_threads`), `stores.py` (`StoreManager`).
  - `tools/`: `toolbox.py` (`Toolbox`, `ToolMetadata`), `common.py` (arXiv and time tools), `summary_tools.py` (`expand_summary`, `summarize_and_store`).
  - `context/`: `usage.py` (token estimate), `summarization.py` (summarise, offload, mark conversations), `sample_data.py` (demo conversation).
  - `rendering.py`: ANSI terminal rendering of memory blocks.
  - `prompts/`: every LLM prompt, as `.md` (static, `load_prompt(name)`) or `.jinja2` (dynamic, `render_prompt(name, **vars)`). Templates are named after the function that uses them.

### Memory stores

Seven Oracle tables, named in `L5.py` (`ALL_TABLES`):

- SQL tables: `CONVERSATIONAL_MEMORY` (per `thread_id`, rows get marked with a `summary_id` once summarized) and `TOOL_LOG_MEMORY` (full tool outputs).
- `OracleVS` vector stores: `SEMANTIC_MEMORY` (knowledge base, arXiv paper chunks), `WORKFLOW_MEMORY` (past tool-step sequences), `TOOLBOX_MEMORY` (tool descriptions for semantic tool retrieval), `ENTITY_MEMORY` (LLM-extracted entities), `SUMMARY_MEMORY`.

`StoreManager` constructs the stores. `MemoryManager` exposes paired `read_*` / `write_*` methods per store. The `read_*` methods return raw rows or documents. `context/segments.py` renders them through `prompts/*_segment.jinja2` into the markdown segments (`## Conversation Memory`, etc.) that the system prompt refers to by name, so changing a heading means updating `prompts/agent_system.md` and `offload_to_summary` too.

### Agent turn (`call_agent`)

1. Assemble a partitioned context with `build_memory_context` (`context/segments.py`).
2. Estimate tokens (`len // 4` against `MODEL_TOKEN_LIMITS`). Above 80%, `offload_to_summary` compresses into `SUMMARY_MEMORY` and leaves summary IDs in context. The agent can call `expand_summary(id)` to recover detail.
3. Tools are retrieved semantically per query (`memory_manager.read_toolbox(query, k=5)`), not passed in full. A tool only reaches the model if its stored description matches the query.
4. OpenAI Chat Completions loop (`gpt-5-mini`, max 10 iterations). Tool results over 3000 chars are truncated for the LLM. The full output goes to `TOOL_LOG_MEMORY`.
5. Afterwards the workflow, entities and assistant reply are written back to memory.

### Tools

Defined as closures inside `register_common_tools` / `register_summary_tools` in `research_assistant/tools/common.py` and `summary_tools.py` and registered with `Toolbox.register_tool` (decorator, optional LLM docstring augmentation). Registration skips the DB write when a tool name already exists in `TOOLBOX_MEMORY`, so changing a tool's docstring has no effect on retrieval until its row is removed (or `--reset`). Execution looks up the in-memory callable via `toolbox._tools_by_name`.

## Gotchas

- Oracle credentials are hardcoded, not read from `.env`. `setup_oracle_database()` connects as `system` with its default admin password and creates user `VECTOR` / `VectorPwd_2025`, which `L5.py` passes explicitly. `ORACLE_PASSWORD` in `.env` must match the admin default in `research_assistant/db/setup.py` or setup fails.
- `setup_oracle_database()` runs on every start and is meant to be idempotent. It picks an ASSM tablespace to avoid ORA-43853 on vector indexes.
- Embeddings run locally via `sentence-transformers/paraphrase-mpnet-base-v2` (HuggingFace), which pulls torch.

## Prompts

Prompts templates are all centralized in a single place. Static prompts are written a markdown. When dynamic, we use jinja2 templates with these settings:
```python
from jinja2 import Environment, FileSystemLoader, StrictUndefined

environment = Environment(
    loader=FileSystemLoader(PROMPT_DIR),
    undefined=StrictUndefined,
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)
```
Jinja2 is useful when you have loops, optional sections, and reusable templates. For simple variable substitution, it might be unnecessary.
It can also introduce ambiguity when a prompt contains literal template syntax, such as JSON examples with braces, or when another system uses its own templating language.
For prompts containing structured data, I recommend explicitly serializing the data in Python rather than constructing JSON manually inside a template as for example:

```python
import json

document_payload = json.dumps(
    {
        "title": title,
        "content": document,
        "metadata": metadata,
    },
    ensure_ascii=False,
)
```

You can then inject that serialized payload into a clearly delimited section of your prompt.



## CUDA / torch pin

`pyproject.toml` declares `torch<=2.7.1` explicitly with `[tool.uv.sources]` pointing at the cu118 index. This machine's driver caps at CUDA 11.8, and torch arrives transitively through sentence-transformers, so the explicit declaration is what makes the pin apply. When changing dependencies, run `uv lock` first and verify before `uv sync`:

```bash
grep -A2 '^name = "torch"' uv.lock   # source must be download.pytorch.org/whl/cu118
grep -cE 'cu1[23]' uv.lock           # must be 0
```
