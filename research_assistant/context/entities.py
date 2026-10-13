import json as json_lib

from research_assistant.prompts import render_prompt


def extract_entities(text: str, llm_client) -> list[dict]:
    """Use LLM to extract entities (people, places, systems) from text."""
    if not text or len(text.strip()) < 5:
        return []

    prompt = render_prompt("extract_entities", text=text[:500])

    try:
        response = llm_client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=2000,
        )
        result = response.choices[0].message.content.strip()

        # Extract JSON array from response
        start, end = result.find("["), result.rfind("]")
        if start == -1 or end == -1:
            return []

        parsed = json_lib.loads(result[start : end + 1])
        return [
            {
                "name": e["name"],
                "type": e.get("type", "UNKNOWN"),
                "description": e.get("description", ""),
            }
            for e in parsed
            if isinstance(e, dict) and e.get("name")
        ]
    # Best effort. A malformed LLM reply must not break the agent turn.
    except Exception:  # noqa: BLE001
        return []
