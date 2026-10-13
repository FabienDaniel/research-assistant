import json as json_lib


def augment_docstring(
        llm_client, model: str, docstring: str, source_code: str = ""
) -> str:
    """
    Use LLM to improve and expand a tool's docstring
    by analyzing both the original description and the
    function's source code.
    """
    if not docstring.strip() and not source_code.strip():
        return "No description provided."

    code_section = ""
    if source_code.strip():
        code_section = (
            "\n\nFunction source code:\n"
            f"```python\n{source_code}\n```"
        )

    prompt = (
        "You are a technical writer. "
        "Analyze the function's source code and its "
        "original docstring, then produce a richer, "
        "more detailed description. Include:\n"
        "1. A clear one-line summary\n"
        "2. What the function does step by step\n"
        "3. When an agent should call this function\n"
        "4. Important notes or caveats\n\n"
        f"Original docstring:\n{docstring}"
        f"{code_section}\n\n"
        "Return ONLY the improved docstring, "
        "no other text."
    )

    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=2000,
    )

    return response.choices[0].message.content.strip()


def generate_queries(llm_client, model: str, docstring: str, num_queries: int = 5) -> list[str]:
    """
    Generate synthetic example queries that would lead to using this tool.
    """
    prompt = f"""Based on the following tool description,
            generate {num_queries} diverse example queries
            that a user might ask when they need this tool. Make them natural and varied.

            Tool description:
            {docstring}

            Return ONLY a JSON array of strings, like: ["query1", "query2", ...]
        """

    response = llm_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=2000
    )

    try:
        queries = json_lib.loads(response.choices[0].message.content.strip())
        return queries if isinstance(queries, list) else []
    except json_lib.JSONDecodeError:
        # Fallback: extract queries from text
        return [response.choices[0].message.content.strip()]
