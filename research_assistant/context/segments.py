# Segment headings are referenced by name in AGENT_SYSTEM_PROMPT (L5.py) and parsed by
# offload_to_summary. Renaming one means updating both.


def conversation_segment(rows) -> str:
    """Format chronological (role, content, timestamp) rows as the Conversation Memory segment."""
    messages = [f"[{ts.strftime('%H:%M:%S')}] [{role}] {content}" for role, content, ts in rows]
    messages_formatted = '\n'.join(messages)
    if not messages_formatted:
        messages_formatted = "(No unsummarized messages found for this thread.)"
    return f"""## Conversation Memory
### What this memory is
Chronological, unsummarized messages from the current thread. This memory captures user intent, constraints, and commitments made in recent turns.
### How you should leverage it
- Preserve continuity with prior decisions, terminology, and user preferences.
- Resolve references like "that", "previous step", or "the paper above" using earlier turns.
- If older context conflicts with newer user instructions, prioritize the latest user direction.
### Retrieved messages

{messages_formatted}"""


def knowledge_base_segment(docs) -> str:
    """Format knowledge-base search results as the Knowledge Base Memory segment."""
    content = "\n".join([doc.page_content for doc in docs])
    if not content:
        content = "(No relevant knowledge base passages found.)"
    return f"""## Knowledge Base Memory
### What this memory is
Retrieved background documents and previously ingested reference material relevant to the current query.
### How you should leverage it
- Ground responses in these passages when making factual or technical claims.
- Prefer concrete details from this memory over unsupported assumptions.
- If evidence is missing or ambiguous, state uncertainty and request clarification or additional retrieval.
### Retrieved passages

{content}"""


def workflow_segment(docs) -> str:
    """Format workflow search results as the Workflow Memory segment."""
    if not docs:
        return """## Workflow Memory
### What this memory is
Past task trajectories that include query context, ordered steps taken, and prior outcomes.
### How you should leverage it
- Use these workflows as reusable execution patterns for planning and tool orchestration.
- Adapt step sequences to the current task rather than copying blindly.
- Reuse successful patterns first, then adjust when task scope or constraints differ.
### Retrieved workflows
(No relevant workflows found.)"""
    content = "\n---\n".join([doc.page_content for doc in docs])
    return f"""## Workflow Memory
### What this memory is
Past task trajectories that include query context, ordered steps taken, and prior outcomes.
### How you should leverage it
- Use these workflows as reusable execution patterns for planning and tool orchestration.
- Adapt step sequences to the current task rather than copying blindly.
- Reuse successful patterns first, then adjust when task scope or constraints differ.
### Retrieved workflows

{content}"""


def entity_segment(docs) -> str:
    """Format entity search results as the Entity Memory segment."""
    if not docs:
        return """## Entity Memory
### What this memory is
Entity-level context such as people, organizations, systems, tools, and other named items previously identified in conversations or documents.
### How you should leverage it
- Use entities to disambiguate references and maintain consistent naming.
- Preserve important attributes (roles, relationships, descriptions) across turns.
- Personalize and contextualize responses using relevant known entities.
### Retrieved entities
(No entities found.)"""

    entities = [f"• {doc.metadata.get('name', '?')}: {doc.metadata.get('description', '')}"
                for doc in docs if hasattr(doc, 'metadata')]
    entities_formatted = '\n'.join(entities)
    return f"""## Entity Memory
### What this memory is
Entity-level context such as people, organizations, systems, tools, and other named items previously identified in conversations or documents.
### How you should leverage it
- Use entities to disambiguate references and maintain consistent naming.
- Preserve important attributes (roles, relationships, descriptions) across turns.
- Personalize and contextualize responses using relevant known entities.
### Retrieved entities

{entities_formatted}"""


def summary_segment(docs, thread_id: str | None = None) -> str:
    """Format summary search results as the Summary Memory segment (IDs + descriptions only)."""
    if not docs:
        scope_note = (
            f"(No summaries available for thread {thread_id}.)"
            if thread_id is not None
            else "(No summaries available.)"
        )
        return """## Summary Memory
### What this memory is
Compressed snapshots of older conversation windows preserved to retain long-range context.
### How you should leverage it
- Use summaries to maintain continuity when full historical messages are not in the active context window.
- Call expand_summary(id) before depending on exact quotes, fine-grained details, or step-by-step chronology.
### Available summaries
""" + scope_note

    lines = [
        "## Summary Memory",
        "### What this memory is",
        "Compressed snapshots of older conversation windows preserved to retain long-range context.",
        "### How you should leverage it",
        "- Use summaries to maintain continuity when full historical messages are not in the active context window.",
        "- Call expand_summary(id) before depending on exact quotes, fine-grained details, or step-by-step chronology.",
        "### Available summaries",
        "Use expand_summary(id) to retrieve the detailed underlying conversation."
    ]
    if thread_id is not None:
        lines.append(f"Scope: thread_id = {thread_id}")
    for doc in docs:
        sid = doc.metadata.get('id', '?')
        desc = doc.metadata.get('description', 'No description')
        lines.append(f"  • [ID: {sid}] {desc}")
    return "\n".join(lines)


def expanded_conversations_segment(summary_id: str, rows) -> str:
    """Format the original (id, role, content, timestamp) rows behind a summary, oldest first."""
    if not rows:
        return f"No conversations found for summary_id: {summary_id}"

    lines = [f"## Expanded Conversations for Summary ID: {summary_id}"]
    lines.append(f"Total messages: {len(rows)}\n")

    for msg_id, role, content, timestamp in rows:
        ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S') if timestamp else "Unknown"
        lines.append(f"[{ts_str}] [{role.upper()}]")
        lines.append(f"{content}")
        lines.append("")  # Empty line between messages

    return "\n".join(lines)


def build_memory_context(memory_manager, query: str, thread_id: str) -> str:
    """Assemble every memory segment for one agent turn, in the order the system prompt lists them."""
    segments = [
        conversation_segment(memory_manager.read_conversational_memory(thread_id)),
        knowledge_base_segment(memory_manager.read_knowledge_base(query)),
        workflow_segment(memory_manager.read_workflow(query)),
        entity_segment(memory_manager.read_entity(query)),
        summary_segment(memory_manager.read_summary_context(query, thread_id=thread_id), thread_id=thread_id),
    ]
    return "".join(segment + "\n\n" for segment in segments)
