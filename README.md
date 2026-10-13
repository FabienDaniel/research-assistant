The **agentic research assistant** helps users investigate complex topics over multiple sessions. The assistant must remember prior findings, source credibility, and user preferences so it can deliver consistent, context-aware answers without repeating the same discovery work each time.


# launch

The oracle database is managed via the docker-compose.yml. The status of the container 'oracle-free' can be seen through

```bash
docker logs -f oracle-free
```

then launch the upp via

```bash
uv run python L5.py
```

