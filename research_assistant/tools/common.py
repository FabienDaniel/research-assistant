import json as json_lib
from datetime import datetime

from research_assistant.tools.summary_tools import register_summary_tools


def register_common_tools(
        toolbox,
        memory_manager,
        knowledge_base_table: str = "SEMANTIC_MEMORY",
        include_summary_tools: bool = True,
):
    """
    Register common tools that should be available across all lessons.

    This function creates and registers tools for:
    - ArXiv paper search
    - ArXiv paper fetch and save to knowledge base
    - Get current time
    - Conversation summarization and summary expansion (optional)

        Args:
        toolbox: The Toolbox instance to register tools with
        memory_manager: The MemoryManager instance for storing data
        knowledge_base_table: Name of the knowledge base table
        include_summary_tools: If True, also registers summary tools

    Returns:
        dict: Dictionary of registered tool names to their functions
    """
    import urllib.request
    import arxiv
    import pymupdf
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from datetime import timezone

    # These tools talk to arxiv.py directly rather than through langchain_community's
    # ArxivRetriever/ArxivLoader. That integration calls Search.results() and
    # Result.download_pdf(), both removed in arxiv 4.x, and langchain-community is
    # sunset so it will not be fixed upstream. arxiv 4.x has no download API at all,
    # hence the manual PDF fetch below.
    arxiv_client = arxiv.Client()

    # arXiv rejects requests that do not identify themselves.
    arxiv_user_agent = "research-assistant/0.1 (langchain_oracledb demo)"

    # Define the tools
    def arxiv_search_candidates(query: str, k: int = 5) -> str:
        """
        Search arXiv and return a JSON list of candidate papers with IDs + metadata.

        Output schema (JSON string):
        [
          {
            "arxiv_id": "2310.08560v2",
            "entry_id": "http://arxiv.org/abs/2310.08560v2",
            "title": "...",
            "authors": "...",
            "published": "2024-02-12",
            "abstract": "..."
          },
          ...
        ]
        """
        results = arxiv_client.results(arxiv.Search(query=query, max_results=k))
        candidates = []
        for result in results:
            candidates.append({
                "arxiv_id": result.get_short_id(),
                "entry_id": result.entry_id,
                "title": result.title,
                "authors": ", ".join(author.name for author in result.authors),
                "published": str(result.published),
                "abstract": (result.summary or "")[:2500],
            })
        return json_lib.dumps(candidates, ensure_ascii=False, indent=2)

    def fetch_and_save_paper_to_kb_db(
            arxiv_id: str,
            chunk_size: int = 1500,
            chunk_overlap: int = 200,
    ) -> str:
        """
        Fetch full arXiv paper text (PDF -> text) and store it into the OracleVS
        knowledge base table as chunked records.
        """
        try:
            result = next(iter(arxiv_client.results(arxiv.Search(id_list=[arxiv_id]))))
        except StopIteration:
            return f"No documents found for arXiv id: {arxiv_id}"

        title = result.title or f"arXiv {arxiv_id}"
        entry_id = result.entry_id or ""
        published = result.published or ""
        authors = ", ".join(author.name for author in result.authors)

        request = urllib.request.Request(
            result.pdf_url, headers={"User-Agent": arxiv_user_agent}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            pdf_bytes = response.read()

        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
            full_text = "\n".join(page.get_text() for page in pdf)

        if not full_text.strip():
            return f"Loaded arXiv {arxiv_id} but extracted empty text (PDF parsing issue)."

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_text(full_text)

        ts_utc = datetime.now(timezone.utc).isoformat()
        metadatas = []
        for i in range(len(chunks)):
            metadatas.append({
                "source": "arxiv",
                "arxiv_id": arxiv_id,
                "title": title,
                "entry_id": entry_id,
                "published": str(published),
                "authors": str(authors),
                "chunk_id": i,
                "num_chunks": len(chunks),
                "ingested_ts_utc": ts_utc,
            })

        memory_manager.write_knowledge_base(chunks, metadatas)

        return (
            f"Saved arXiv {arxiv_id} to {knowledge_base_table}: "
            f"{len(chunks)} chunks (title: {title})."
        )

    def get_current_time(detailed: bool = False) -> str:
        """
        Returns the current time.

        Args:
            detailed: If True, returns detailed format with microseconds

        Returns:
            str: Current time as formatted string
        """
        if detailed:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        else:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Register the tools with the toolbox
    toolbox.register_tool(arxiv_search_candidates, augment=False)
    toolbox.register_tool(fetch_and_save_paper_to_kb_db, augment=True)
    toolbox.register_tool(get_current_time, augment=False)

    # Store references so they can be called directly
    registered_tools = {
        "arxiv_search_candidates": arxiv_search_candidates,
        "fetch_and_save_paper_to_kb_db": fetch_and_save_paper_to_kb_db,
        "get_current_time": get_current_time,
    }

    # Optionally include summary tools in the common toolset
    if include_summary_tools:
        summary_tools = register_summary_tools(
            toolbox=toolbox,
            memory_manager=memory_manager,
            llm_client=toolbox.llm_client,
        )
        registered_tools.update(summary_tools)

    print(f"✅ Registered {len(registered_tools)} common tools: {list(registered_tools.keys())}")

    return registered_tools
