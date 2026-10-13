import uuid

from research_assistant.prompts import render_prompt


def summarise_context_window(
        content: str,
        memory_manager,
        llm_client,
        model: str = "gpt-5",
        thread_id: str | None = None,
) -> dict:
    """
    Summarise content using LLM and store in summary memory.
    Captures technical details, emotional context, and entity information.
    """
    cleaned = (content or "").strip()
    if not cleaned:
        return {"status": "nothing_to_summarize"}

    def _message_text(resp) -> str:
        msg = resp.choices[0].message
        payload = getattr(msg, "content", None)
        if isinstance(payload, str):
            return payload.strip()
        if isinstance(payload, list):
            parts: list[str] = []
            for item in payload:
                if isinstance(item, dict):
                    txt = item.get("text")
                    if isinstance(txt, str) and txt.strip():
                        parts.append(txt.strip())
            return "\n".join(parts).strip()
        return ""

    def _fallback_description(summary_text: str, source_text: str) -> str:
        """Create a specific, non-generic summary label when model labeling is weak."""
        for raw_line in (summary_text or source_text).splitlines():
            line = raw_line.strip().strip("-•* ").strip()
            if not line or line.startswith("###"):
                continue
            words = [w for w in line.split() if w]
            if len(words) < 4:
                continue
            label = " ".join(words[:12]).strip(" ,.;:")
            if label:
                return label
        return "Recent thread context, decisions, and open actions"

    summary_prompt = render_prompt("summarize_conversation", conversation=cleaned[:6000])

    response = llm_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": summary_prompt}],
        max_completion_tokens=4000
    )
    summary = _message_text(response)

    # Retry once with a simpler prompt if the model returns empty content.
    if not summary:
        retry_prompt = render_prompt("summarize_conversation_retry", conversation=cleaned[:6000])
        retry = llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": retry_prompt}],
            max_completion_tokens=4000
        )
        summary = _message_text(retry)

    if not summary:
        # Deterministic fallback so downstream flow never breaks.
        excerpt = cleaned[:500].replace("\n", " ").strip()
        summary = (
            "### Technical Information\n"
            f"{excerpt or '(No content provided.)'}\n\n"
            "### Emotional Context\n"
            "Not available from model output.\n\n"
            "### Entities & References\n"
            "Not available from model output.\n\n"
            "### Action Items & Decisions\n"
            "Not available from model output."
        )

    desc_prompt = render_prompt("summary_label", summary=summary)

    desc_response = llm_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": desc_prompt}],
        max_completion_tokens=2000
    )
    description = _message_text(desc_response).strip().strip('"').strip("'")
    if not description or description.lower() in {
        "conversation summary",
        "summary",
        "chat summary",
        "thread summary",
    }:
        description = _fallback_description(summary, cleaned)

    summary_id = str(uuid.uuid4())[:8]
    memory_manager.write_summary(summary_id, cleaned, summary, description, thread_id=thread_id)

    return {"id": summary_id, "description": description, "summary": summary}


def offload_to_summary(context: str, memory_manager, llm_client, thread_id: str | None = None) -> tuple[
    str, list[dict]]:
    """
    Simple context compaction:
    - If thread_id is provided, summarize unsummarized conversation units for that thread.
    - Otherwise, summarize the provided context string.
    - Return a compact context with summary references that can be expanded on demand.
    """
    raw_context = (context or "").strip()

    if thread_id is not None:
        result = summarize_conversation(thread_id, memory_manager, llm_client)
    else:
        result = summarise_context_window(raw_context, memory_manager, llm_client, thread_id=thread_id)

    if result.get("status") == "nothing_to_summarize":
        return raw_context, []

    summary_ref = f"[Summary ID: {result['id']}] {result['description']}"
    conversation_stub = (
        "## Conversation Memory\n"
        "Older conversation content was summarized to reduce context size.\n"
        "Use Summary Memory references + expand_summary(id) for full detail."
    )

    # Replace only the conversation section, keep other memory segments intact.
    compact_context = raw_context
    if "## Conversation Memory" in compact_context:
        lines = compact_context.splitlines()
        rebuilt: list[str] = []
        in_conversation = False
        inserted_stub = False

        for line in lines:
            if line.startswith("## "):
                heading = line.strip()
                if heading == "## Conversation Memory":
                    in_conversation = True
                    if not inserted_stub:
                        if rebuilt and rebuilt[-1].strip():
                            rebuilt.append("")
                        rebuilt.extend(conversation_stub.splitlines())
                        rebuilt.append("")
                        inserted_stub = True
                    continue
                in_conversation = False

            if not in_conversation:
                rebuilt.append(line)

        compact_context = "\n".join(rebuilt).strip()
    else:
        compact_context = f"{conversation_stub}\n\n{compact_context}".strip()

    # Add summary reference(s).
    if "## Summary Memory" in compact_context:
        compact_context = f"{compact_context}\n{summary_ref}".strip()
    else:
        compact_context = (
            f"{compact_context}\n\n"
            "## Summary Memory\n"
            "Use expand_summary(id) to retrieve full underlying content.\n"
            f"{summary_ref}"
        ).strip()

    return compact_context, [result]


def summarize_conversation(thread_id: str, memory_manager, llm_client) -> dict:
    """
    Summarize all unsummarized messages in a thread and mark them.

    This function:
    1. Reads all unsummarized messages from the current thread
    2. Generates a comprehensive summary capturing technical, emotional, and entity info
    3. Stores the summary in the summary memory store
    4. Marks all conversation messages with the summary_id for later retrieval
    5. Returns the summary info that can be used to start a new context window
    """
    thread_id = str(thread_id)
    with memory_manager.conn.cursor() as cur:
        cur.execute(f"""
            SELECT id, role, content, timestamp
            FROM {memory_manager.conversation_table}
            WHERE thread_id = :thread_id AND summary_id IS NULL
            ORDER BY timestamp ASC
        """, {"thread_id": thread_id})
        rows = cur.fetchall()

    if not rows:
        return {"status": "nothing_to_summarize"}

    message_ids: list[str] = []
    transcript_lines: list[str] = []
    for msg_id, role, content, timestamp in rows:
        message_ids.append(msg_id)
        ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S') if timestamp else "Unknown"
        transcript_lines.append(f"[{ts_str}] [{str(role).upper()}] {content}")

    transcript = "\n".join(transcript_lines).strip()
    if not transcript:
        return {"status": "nothing_to_summarize"}

    result = summarise_context_window(transcript, memory_manager, llm_client, thread_id=thread_id)
    if result.get("status") == "nothing_to_summarize":
        return result

    summary_id = result["id"]
    with memory_manager.conn.cursor() as cur:
        cur.executemany(f"""
            UPDATE {memory_manager.conversation_table}
            SET summary_id = :summary_id
            WHERE id = :id AND summary_id IS NULL
        """, [{"summary_id": summary_id, "id": msg_id} for msg_id in message_ids])
    memory_manager.conn.commit()

    result["num_messages_summarized"] = len(message_ids)

    print(f"✅ Conversation summarized: [Summary ID: {summary_id}]")
    print(f"   Description: {result['description']}")
    print(f"   Messages marked summarized: {len(message_ids)}")

    return result
