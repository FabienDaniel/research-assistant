from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_oracledb.vectorstores import OracleVS

from research_assistant.db.schema import (
    CONVERSATIONAL_TABLE,
    ENTITY_TABLE,
    KNOWLEDGE_BASE_TABLE,
    SUMMARY_TABLE,
    TOOL_LOG_TABLE,
    TOOLBOX_TABLE,
    WORKFLOW_TABLE,
    create_conversational_history_table,
    create_tool_log_table,
)
from research_assistant.memory.manager import MemoryManager


def build_memory_manager(conn, embedding_model) -> MemoryManager:
    """Create any missing memory tables and return a MemoryManager over them."""

    def vector_store(table_name: str) -> OracleVS:
        return OracleVS(
            client=conn,
            embedding_function=embedding_model,
            table_name=table_name,
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        )

    return MemoryManager(
        conn=conn,
        conversation_table=create_conversational_history_table(conn, CONVERSATIONAL_TABLE),
        knowledge_base_vs=vector_store(KNOWLEDGE_BASE_TABLE),
        workflow_vs=vector_store(WORKFLOW_TABLE),
        toolbox_vs=vector_store(TOOLBOX_TABLE),
        entity_vs=vector_store(ENTITY_TABLE),
        summary_vs=vector_store(SUMMARY_TABLE),
        tool_log_table=create_tool_log_table(conn, TOOL_LOG_TABLE),
    )
