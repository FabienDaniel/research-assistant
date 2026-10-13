from research_assistant.context.segments import expanded_conversations_segment
from research_assistant.context.summarization import summarise_context_window, summarize_conversation


def register_summary_tools(toolbox, memory_manager, llm_client):
    """
    Register summary-related tools with the toolbox.

    Args:
        toolbox: The Toolbox instance to register tools with
        memory_manager: MemoryManager instance for memory operations
        llm_client: OpenAI client for LLM calls

    Returns:
        dict of registered tool functions
    """

    def expand_summary(summary_id: str, thread_id: str = None) -> str:
        """
        Expand a summary reference to retrieve the original conversations.

        Use when you need more details from a [Summary ID: xxx] reference.
        Returns all original messages that were summarized, in chronological order with timestamps.
        """
        summary_text = memory_manager.read_summary_memory(summary_id, thread_id=thread_id)
        original_conversations = expanded_conversations_segment(
            summary_id, memory_manager.read_conversations_by_summary_id(summary_id)
        )

        return f"""## Summary Context
{summary_text}

{original_conversations}"""

    def summarize_and_store(text: str = "", thread_id: str = None) -> str:
        """
        Summarize content and store it in Summary Memory.

        Preferred usage for agent continuity:
        - Provide `thread_id` to summarize unsummarized conversation units and mark them.

        Alternate usage:
        - Provide `text` to summarize arbitrary content.
        """
        if thread_id:
            result = summarize_conversation(thread_id, memory_manager, llm_client)
            if result.get("status") == "nothing_to_summarize":
                return f"No unsummarized messages found for thread {thread_id}."
            return f"Stored as [Summary ID: {result['id']}] {result['description']}"

        text = (text or "").strip()
        if not text:
            return "Provide `thread_id` (preferred) or non-empty `text` to summarize."

        result = summarise_context_window(text, memory_manager, llm_client)
        if result.get("status") == "nothing_to_summarize":
            return "No content to summarize."
        return f"Stored as [Summary ID: {result['id']}] {result['description']}"

    # Register with toolbox
    toolbox.register_tool(expand_summary, augment=True)
    toolbox.register_tool(summarize_and_store, augment=True)

    registered_tools = {
        "expand_summary": expand_summary,
        "summarize_and_store": summarize_and_store,
    }

    print(f"✅ Registered {len(registered_tools)} summary tools: {list(registered_tools.keys())}")

    return registered_tools
