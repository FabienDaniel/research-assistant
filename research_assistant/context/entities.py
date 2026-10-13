import json as json_lib


def extract_entities(text: str, llm_client) -> list[dict]:
    """Use LLM to extract entities (people, places, systems) from text."""
    if not text or len(text.strip()) < 5:
        return []

    prompt = f'''Extract entities from: "{text[:500]}"
Return JSON: [{{"name": "X", "type": "PERSON|PLACE|SYSTEM", "description": "brief"}}]
If none: []'''

    try:
        response = llm_client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=2000
        )
        result = response.choices[0].message.content.strip()

        # Extract JSON array from response
        start, end = result.find("["), result.rfind("]")
        if start == -1 or end == -1:
            return []

        parsed = json_lib.loads(result[start:end + 1])
        return [{"name": e["name"], "type": e.get("type", "UNKNOWN"), "description": e.get("description", "")}
                for e in parsed if isinstance(e, dict) and e.get("name")]
    except:
        return []
