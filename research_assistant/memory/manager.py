import json as json_lib
from datetime import datetime

import oracledb


def list_threads(conn, table_name):
    """
    List conversation threads with their message count and last activity.

    Counts summarized messages too, so a compacted thread stays visible.
    Returns a list of (thread_id, message_count, last_timestamp), most recent first.
    """
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT thread_id, COUNT(*), MAX(timestamp)
            FROM {table_name}
            GROUP BY thread_id
            ORDER BY MAX(timestamp) DESC
        """)
        return cur.fetchall()


class MemoryManager:
    """
    A simplified memory manager for AI agents using Oracle AI Database.

    Manages 7 types of memory:
    - Conversational: Chat history per thread (SQL table)
    - Tool Log: Raw tool execution outputs and metadata (SQL table)
    - Knowledge Base: Searchable documents (Vector store)
    - Workflow: Execution patterns (Vector store)
    - Toolbox: Available tools (Vector store)
    - Entity: People, places, systems (Vector store)
    - Summary: Storing compressed context window
    """

    def __init__(
            self,
            conn,
            conversation_table: str,
            knowledge_base_vs,
            workflow_vs,
            toolbox_vs,
            entity_vs,
            summary_vs,
            tool_log_table: str | None = None
    ):
        self.conn = conn
        self.conversation_table = conversation_table
        self.knowledge_base_vs = knowledge_base_vs
        self.workflow_vs = workflow_vs
        self.toolbox_vs = toolbox_vs
        self.entity_vs = entity_vs
        self.summary_vs = summary_vs
        self.tool_log_table = tool_log_table

    # ==================== CONVERSATIONAL MEMORY (SQL) ====================

    def write_conversational_memory(self, content: str, role: str, thread_id: str) -> str:
        """Store a message in conversation history."""
        thread_id = str(thread_id)
        with self.conn.cursor() as cur:
            id_var = cur.var(str)
            cur.execute(f"""
                INSERT INTO {self.conversation_table} (thread_id, role, content, metadata, timestamp)
                VALUES (:thread_id, :role, :content, :metadata, CURRENT_TIMESTAMP)
                RETURNING id INTO :id
            """, {"thread_id": thread_id, "role": role, "content": content, "metadata": "{}", "id": id_var})
            record_id = id_var.getvalue()[0] if id_var.getvalue() else None
        self.conn.commit()
        return record_id

    def read_conversational_memory(self, thread_id: str, limit: int = 10) -> list[tuple]:
        """Read recent unsummarized (role, content, timestamp) rows for a thread, oldest first."""
        thread_id = str(thread_id)
        with self.conn.cursor() as cur:
            # Newest first so a long-lived thread keeps its recent turns, then reversed
            # below to return them chronologically.
            cur.execute(f"""
                SELECT role, content, timestamp FROM {self.conversation_table}
                WHERE thread_id = :thread_id AND summary_id IS NULL
                ORDER BY timestamp DESC
                FETCH FIRST :limit ROWS ONLY
            """, {"thread_id": thread_id, "limit": limit})
            return list(reversed(cur.fetchall()))

    def mark_as_summarized(self, thread_id: str, summary_id: str):
        """Mark all unsummarized messages in a thread as summarized."""
        thread_id = str(thread_id)
        with self.conn.cursor() as cur:
            cur.execute(f"""
                UPDATE {self.conversation_table}
                SET summary_id = :summary_id
                WHERE thread_id = :thread_id AND summary_id IS NULL
            """, {"summary_id": summary_id, "thread_id": thread_id})
        self.conn.commit()
        print(f"  📦 Marked messages as summarized (summary_id: {summary_id})")

    # ==================== TOOL LOG MEMORY (SQL) ====================

    def write_tool_log(
            self,
            thread_id: str,
            tool_name: str,
            tool_args,
            result: str,
            status: str = "success",
            tool_call_id: str | None = None,
            error_message: str | None = None,
            metadata: dict | None = None,
    ) -> str | None:
        """Persist raw tool execution logs for auditing and just-in-time retrieval."""
        if not self.tool_log_table:
            return None

        thread_id = str(thread_id)

        if isinstance(tool_args, (dict, list)):
            tool_args_str = json_lib.dumps(tool_args, ensure_ascii=False)
        else:
            tool_args_str = "" if tool_args is None else str(tool_args)

        result_str = "" if result is None else str(result)
        # Oracle VARCHAR2(2000) may be byte-limited; truncate preview by UTF-8 bytes.
        preview = result_str.encode("utf-8")[:2000].decode("utf-8", errors="ignore")

        metadata_str = json_lib.dumps(metadata, ensure_ascii=False) if metadata else "{}"

        with self.conn.cursor() as cur:
            id_var = cur.var(str)
            cur.execute(f"""
                INSERT INTO {self.tool_log_table}
                    (thread_id, tool_call_id, tool_name, tool_args, result, result_preview, status, error_message, metadata, timestamp)
                VALUES
                    (:thread_id, :tool_call_id, :tool_name, :tool_args, :result, :result_preview, :status, :error_message, :metadata, CURRENT_TIMESTAMP)
                RETURNING id INTO :id
            """, {
                "thread_id": thread_id,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "tool_args": tool_args_str,
                "result": result_str,
                "result_preview": preview,
                "status": status,
                "error_message": error_message,
                "metadata": metadata_str,
                "id": id_var
            })
            log_id = id_var.getvalue()[0] if id_var.getvalue() else None

        self.conn.commit()
        return log_id

    def read_tool_logs(self, thread_id: str, limit: int = 20) -> list[dict]:
        """Read recent tool logs for a thread, newest first."""
        if not self.tool_log_table:
            return []

        thread_id = str(thread_id)
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT id, tool_call_id, tool_name, tool_args, result_preview, status, error_message, metadata, timestamp
                FROM {self.tool_log_table}
                WHERE thread_id = :thread_id
                ORDER BY timestamp DESC
                FETCH FIRST :limit ROWS ONLY
            """, {"thread_id": thread_id, "limit": limit})
            rows = cur.fetchall()

        logs = []
        for log_id, tool_call_id, tool_name, tool_args, result_preview, status, error_message, metadata, ts in rows:
            logs.append({
                "id": log_id,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "result_preview": result_preview,
                "status": status,
                "error_message": error_message,
                "metadata": metadata,
                "timestamp": ts.isoformat() if ts else None,
            })
        return logs

    # ==================== KNOWLEDGE BASE (Vector Store) ====================

    def write_knowledge_base(self, text: str | list[str], metadata: dict | list[dict]):
        """
        Store knowledge-base content with metadata.

        Supports:
        - Single record: text=str, metadata=dict
        - Batch insert: text=list[str], metadata=list[dict]
        """
        if isinstance(text, list):
            texts = [str(t) for t in text]
            if isinstance(metadata, list):
                metadatas = metadata
            else:
                metadatas = [metadata for _ in texts]

            if len(texts) != len(metadatas):
                raise ValueError(
                    f"Knowledge-base batch length mismatch: {len(texts)} texts vs {len(metadatas)} metadata rows"
                )
            self.knowledge_base_vs.add_texts(texts, metadatas)
            return

        self.knowledge_base_vs.add_texts([str(text)], [metadata if isinstance(metadata, dict) else {}])

    def read_knowledge_base(self, query: str, k: int = 3) -> list:
        """Search knowledge base for relevant passages."""
        return self.knowledge_base_vs.similarity_search(query, k=k)

    # ==================== WORKFLOW (Vector Store) ====================

    def write_workflow(self, query: str, steps: list, final_answer: str, success: bool = True):
        """Store a completed workflow pattern for future reference."""
        # Format steps as text
        steps_text = "\n".join([f"Step {i + 1}: {s}" for i, s in enumerate(steps)])
        text = f"Query: {query}\nSteps:\n{steps_text}\nAnswer: {final_answer[:200]}"

        metadata = {
            "query": query,
            "success": success,
            "num_steps": len(steps),
            "timestamp": datetime.now().isoformat()
        }
        self.workflow_vs.add_texts([text], [metadata])

    def read_workflow(self, query: str, k: int = 3) -> list:
        """Search for similar past workflows with at least 1 step."""
        # Filter to only include workflows that have steps (num_steps > 0)
        return self.workflow_vs.similarity_search(
            query,
            k=k,
            filter={"num_steps": {"$gt": 0}}
        )

    # ==================== TOOLBOX (Vector Store) ====================

    def write_toolbox(self, text: str, metadata: dict):
        """Store a tool definition in the toolbox."""
        self.toolbox_vs.add_texts([text], [metadata])

    def toolbox_has(self, tool_name: str) -> bool:
        """Check if a tool with the given name already exists in the toolbox store."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"SELECT COUNT(*) FROM {self.toolbox_vs.table_name} "
                    f"WHERE JSON_VALUE(metadata, '$.name') = :name",
                    {"name": tool_name},
                )
                return cur.fetchone()[0] > 0
        except oracledb.Error:
            return False

    def read_toolbox(self, query: str, k: int = 3) -> list[dict]:
        """Find relevant tool metadata, one entry per tool name."""
        results = self.toolbox_vs.similarity_search(query, k=k)
        tools = []
        seen_tool_names: set[str] = set()
        for doc in results:
            meta = doc.metadata
            tool_name = meta.get("name", "tool")
            if tool_name in seen_tool_names:
                continue
            seen_tool_names.add(tool_name)
            tools.append(meta)
        return tools

    # ==================== ENTITY (Vector Store) ====================

    def write_entity(self, name: str, entity_type: str, description: str):
        """Store a single entity."""
        self.entity_vs.add_texts(
            [f"{name} ({entity_type}): {description}"],
            [{"name": name, "type": entity_type, "description": description}]
        )

    def read_entity(self, query: str, k: int = 5) -> list:
        """Search for relevant entities."""
        return self.entity_vs.similarity_search(query, k=k)

    # ==================== SUMMARY (Vector Store) ====================

    def write_summary(
            self,
            summary_id: str,
            full_content: str,
            summary: str,
            description: str,
            thread_id: str | None = None,
    ):
        """Store a summary with its original content."""
        metadata = {
            "id": summary_id,
            "full_content": full_content,
            "summary": summary,
            "description": description,
        }
        if thread_id is not None:
            metadata["thread_id"] = str(thread_id)
        self.summary_vs.add_texts(
            [f"{summary_id}: {description}"],
            [metadata]
        )
        return summary_id

    def read_summary_memory(self, summary_id: str, thread_id: str | None = None) -> str:
        """Retrieve a specific summary by ID (just-in-time retrieval)."""
        filters = {"id": summary_id}
        if thread_id is not None:
            filters["thread_id"] = str(thread_id)

        results = self.summary_vs.similarity_search(
            summary_id,
            k=5,
            filter=filters
        )
        if not results:
            if thread_id is not None:
                return f"Summary {summary_id} not found for thread {thread_id}."
            return f"Summary {summary_id} not found."
        doc = results[0]
        return doc.metadata.get('summary', 'No summary content.')

    def read_summary_context(self, query: str = "", k: int = 10, thread_id: str | None = None) -> list:
        """Search available summaries, scoped to a thread when one is given."""
        filters = None
        if thread_id is not None:
            filters = {"thread_id": str(thread_id)}
        return self.summary_vs.similarity_search(query or "summary", k=k, filter=filters)

    def read_conversations_by_summary_id(self, summary_id: str) -> list[tuple]:
        """Read the original (id, role, content, timestamp) rows behind a summary, oldest first."""
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT id, role, content, timestamp 
                FROM {self.conversation_table}
                WHERE summary_id = :summary_id
                ORDER BY timestamp ASC
            """, {"summary_id": summary_id})
            return cur.fetchall()
