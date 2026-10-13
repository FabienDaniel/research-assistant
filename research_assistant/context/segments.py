# Segment headings are referenced by name in prompts/agent_system.md and parsed by
# offload_to_summary. Renaming one means updating both.
from research_assistant.prompts import render_prompt


def conversation_segment(rows) -> str:
    """Format chronological (role, content, timestamp) rows as the Conversation Memory segment."""
    return render_prompt("conversation_segment", rows=rows)


def knowledge_base_segment(docs) -> str:
    """Format knowledge-base search results as the Knowledge Base Memory segment."""
    return render_prompt("knowledge_base_segment", docs=docs)


def workflow_segment(docs) -> str:
    """Format workflow search results as the Workflow Memory segment."""
    return render_prompt("workflow_segment", docs=docs)


def entity_segment(docs) -> str:
    """Format entity search results as the Entity Memory segment."""
    return render_prompt("entity_segment", docs=docs)


def summary_segment(docs, thread_id: str | None = None) -> str:
    """Format summary search results as the Summary Memory segment (IDs + descriptions only)."""
    return render_prompt("summary_segment", docs=docs, thread_id=thread_id)


def expanded_conversations_segment(summary_id: str, rows) -> str:
    """Format the original (id, role, content, timestamp) rows behind a summary, oldest first."""
    return render_prompt("expanded_conversations_segment", summary_id=summary_id, rows=rows)


def build_memory_context(memory_manager, query: str, thread_id: str) -> str:
    """Assemble every memory segment for one agent turn, in the order the system prompt lists them."""
    segments = [
        conversation_segment(memory_manager.read_conversational_memory(thread_id)),
        knowledge_base_segment(memory_manager.read_knowledge_base(query)),
        workflow_segment(memory_manager.read_workflow(query)),
        entity_segment(memory_manager.read_entity(query)),
        summary_segment(
            memory_manager.read_summary_context(query, thread_id=thread_id),
            thread_id=thread_id,
        ),
    ]
    return "".join(segment + "\n\n" for segment in segments)
