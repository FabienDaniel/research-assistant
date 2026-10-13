from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

PROMPT_DIR = Path(__file__).parent

environment = Environment(
    loader=FileSystemLoader(PROMPT_DIR),
    undefined=StrictUndefined,
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


def load_prompt(name: str) -> str:
    """Return the static prompt stored in `<name>.md`."""
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


def render_prompt(name: str, **variables) -> str:
    """Render the dynamic prompt stored in `<name>.jinja2`."""
    # Block tags closing a loop or branch leave a trailing newline behind.
    return environment.get_template(f"{name}.jinja2").render(**variables).strip()
