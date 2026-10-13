import json as json_lib

from research_assistant.prompts import render_prompt


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

    prompt = render_prompt(
        "augment_docstring", docstring=docstring, source_code=source_code
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
    prompt = render_prompt("generate_queries", docstring=docstring, num_queries=num_queries)

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
