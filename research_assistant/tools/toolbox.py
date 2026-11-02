import inspect
import uuid
from collections.abc import Callable

from pydantic import BaseModel

from research_assistant.tools.augmentation import augment_docstring, generate_queries


class ToolMetadata(BaseModel):
    """Metadata for a registered tool."""
    name: str
    description: str
    signature: str
    parameters: dict
    return_type: str


class Toolbox:
    """
    A toolbox for registering, storing, and retrieving tools with LLM-powered augmentation.

    Tools are stored with embeddings for semantic retrieval, allowing the agent to
    find relevant tools based on natural language queries.
    """

    def __init__(self, memory_manager, llm_client, embedding_function, model: str = "gpt-5"):
        """
        Initialize the Toolbox.

        Args:
            memory_manager: MemoryManager instance for storing tools
            llm_client: LLM client for augmentation
            embedding_function: Embedding function/model for creating embeddings
            model: LLM model name
        """
        self.memory_manager = memory_manager
        self.llm_client = llm_client
        self.embedding_function = embedding_function
        self.model = model
        self._tools: dict[str, Callable] = {}
        self._tools_by_name: dict[str, Callable] = {}

    def _get_embedding(self, text: str) -> list[float]:
        """
        Get the embedding for a text using the configured embedding function.
        """
        if hasattr(self.embedding_function, 'embed_query'):
            return self.embedding_function.embed_query(text)
        elif callable(self.embedding_function):
            return self.embedding_function(text)
        else:
            raise ValueError("embedding_function must be callable or have embed_query method")

    def _get_tool_metadata(self, func: Callable) -> ToolMetadata:
        """
        Extract metadata from a function for storage and retrieval.
        """
        sig = inspect.signature(func)

        # Extract parameter info
        parameters = {}
        for name, param in sig.parameters.items():
            param_info = {"name": name}
            if param.annotation != inspect.Parameter.empty:
                param_info["type"] = str(param.annotation)
            if param.default != inspect.Parameter.empty:
                param_info["default"] = str(param.default)
            parameters[name] = param_info

        # Extract return type
        return_type = "Any"
        if sig.return_annotation != inspect.Signature.empty:
            return_type = str(sig.return_annotation)

        return ToolMetadata(
            name=func.__name__,
            description=func.__doc__ or "No description",
            signature=str(sig),
            parameters=parameters,
            return_type=return_type
        )

    def register_tool(
            self, func: Callable | None = None, augment: bool = False
    ) -> str | Callable:
        """
        Register a function as a tool in the toolbox.

        If a tool with the same name already exists in the database,
        the callable is registered in memory but no duplicate row is
        written to the vector store.
        """

        def decorator(f: Callable) -> str:
            tool_name = f.__name__

            # Deduplication: skip DB write if tool already stored
            if self.memory_manager.toolbox_has(tool_name):
                self._tools_by_name[tool_name] = f
                print(f"  ⏭️  Tool '{tool_name}' already in toolbox (skipping DB write)")
                return tool_name

            docstring = f.__doc__ or ""
            signature = str(inspect.signature(f))
            object_id = uuid.uuid4()
            object_id_str = str(object_id)

            if augment:
                # Use LLM to enhance the tool's discoverability
                try:
                    source_code = inspect.getsource(f)
                except (OSError, TypeError):
                    source_code = ""
                augmented_docstring = augment_docstring(
                    self.llm_client, self.model, docstring, source_code
                )
                queries = generate_queries(self.llm_client, self.model, augmented_docstring)

                # Create rich embedding text combining all information
                embedding_text = f"{f.__name__} {augmented_docstring} {signature} {' '.join(queries)}"
                embedding = self._get_embedding(embedding_text)

                tool_data = self._get_tool_metadata(f)
                tool_data.description = augmented_docstring  # Use augmented description

                tool_dict = {
                    "_id": object_id_str,  # Use string, not UUID object
                    "embedding": embedding,
                    "queries": queries,
                    "augmented": True,
                    **tool_data.model_dump(),
                }
            else:
                # Basic registration without augmentation
                embedding = self._get_embedding(f"{f.__name__} {docstring} {signature}")
                tool_data = self._get_tool_metadata(f)

                tool_dict = {
                    "_id": object_id_str,  # Use string, not UUID object
                    "embedding": embedding,
                    "augmented": False,
                    **tool_data.model_dump(),
                }

            # Store the tool in the toolbox memory for retrieval
            # The embedding enables semantic search to find relevant tools
            self.memory_manager.write_toolbox(
                f"{f.__name__} {docstring} {signature}",
                tool_dict
            )

            # Keep reference to the callable for execution
            self._tools[object_id_str] = f
            self._tools_by_name[f.__name__] = f  # Also store by name for easy lookup
            return object_id_str

        if func is None:
            return decorator
        return decorator(func)

    def execute(self, name: str, args: dict) -> str:
        """Run a registered tool by name. Raises ValueError for an unknown tool."""
        if name not in self._tools_by_name:
            raise ValueError(f"Tool '{name}' not found")
        return str(self._tools_by_name[name](**args) or "Done")

    def schemas_for(self, query: str, k: int = 3) -> list[dict]:
        """Find relevant tools and return OpenAI-compatible schemas."""
        tools = []
        for meta in self.memory_manager.read_toolbox(query, k=k):
            tool_name = meta.get("name", "tool")
            # Extract parameters from metadata and convert to OpenAI format
            stored_params = meta.get("parameters", {})
            properties = {}
            required = []

            for param_name, param_info in stored_params.items():
                # Convert stored param info to OpenAI schema format
                param_type = param_info.get("type", "string")
                # Map Python types to JSON schema types
                type_mapping = {
                    "<class 'str'>": "string",
                    "<class 'int'>": "integer",
                    "<class 'float'>": "number",
                    "<class 'bool'>": "boolean",
                    "str": "string",
                    "int": "integer",
                    "float": "number",
                    "bool": "boolean"
                }
                json_type = type_mapping.get(param_type, "string")
                properties[param_name] = {"type": json_type}

                # If no default, it's required
                if "default" not in param_info:
                    required.append(param_name)

            tools.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": meta.get("description", ""),
                    "parameters": {"type": "object", "properties": properties, "required": required}
                }
            })
        return tools
