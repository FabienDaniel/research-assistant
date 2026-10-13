from langchain_oracledb.retrievers.hybrid_search import OracleVectorizerPreference
from langchain_oracledb.vectorstores import OracleVS


class StoreManager:
    """Manages all stores (vector stores and SQL tables) with getter methods for easy access."""

    def __init__(
        self,
        client,
        embedding_function,
        table_names,
        distance_strategy,
        conversational_table,
        tool_log_table: str | None = None,
    ):
        """
        Initialize all stores.

        Args:
            client: Oracle database connection
            embedding_function: Embedding model to use
            table_names: Dict with keys: knowledge_base, workflow, toolbox, entity, summary
            distance_strategy: Distance strategy for vector search
            conversational_table: Name of the conversational history SQL table
            tool_log_table: Name of the SQL tool log table
        """
        self.client = client
        self.embedding_function = embedding_function
        self.distance_strategy = distance_strategy
        self._conversational_table = conversational_table
        self._tool_log_table = tool_log_table

        # Initialize all vector stores
        self._knowledge_base_vs = OracleVS(
            client=client,
            embedding_function=embedding_function,
            table_name=table_names["knowledge_base"],
            distance_strategy=distance_strategy,
        )

        self._workflow_vs = OracleVS(
            client=client,
            embedding_function=embedding_function,
            table_name=table_names["workflow"],
            distance_strategy=distance_strategy,
        )

        self._toolbox_vs = OracleVS(
            client=client,
            embedding_function=embedding_function,
            table_name=table_names["toolbox"],
            distance_strategy=distance_strategy,
        )

        self._entity_vs = OracleVS(
            client=client,
            embedding_function=embedding_function,
            table_name=table_names["entity"],
            distance_strategy=distance_strategy,
        )

        self._summary_vs = OracleVS(
            client=client,
            embedding_function=embedding_function,
            table_name=table_names["summary"],
            distance_strategy=distance_strategy,
        )

        # Store hybrid search preference for knowledge base (optional)
        self._kb_vectorizer_pref = None

    def get_conversational_table(self):
        """Return the conversational history table name."""
        return self._conversational_table

    def get_tool_log_table(self):
        """Return the tool log table name."""
        return self._tool_log_table

    def get_knowledge_base_store(self):
        """Return the knowledge base vector store."""
        return self._knowledge_base_vs

    def get_workflow_store(self):
        """Return the workflow vector store."""
        return self._workflow_vs

    def get_toolbox_store(self):
        """Return the toolbox vector store."""
        return self._toolbox_vs

    def get_entity_store(self):
        """Return the entity vector store."""
        return self._entity_vs

    def get_summary_store(self):
        """Return the summary vector store."""
        return self._summary_vs

    def setup_hybrid_search(self, preference_name="KB_VECTORIZER_PREF"):
        """
        Set up hybrid search for knowledge base.
        Creates vectorizer preference for hybrid indexing.
        """
        self._kb_vectorizer_pref = OracleVectorizerPreference.create_preference(
            vector_store=self._knowledge_base_vs, preference_name=preference_name
        )
        return self._kb_vectorizer_pref
