import re
import sys

_ANSI_RESET = "\033[0m"
_ANSI_DIM = "\033[2m"
_ANSI_H1 = "\033[1;35m"
_ANSI_H2 = "\033[1;36m"
_ANSI_H3 = "\033[2;33m"
_ANSI_ROLE = {"user": "\033[32m", "assistant": "\033[34m", "tool": "\033[35m"}

# Subsections carrying retrieved content. Everything under the other subsections is
# the static "what this memory is" preamble, identical on every turn.
_CONTENT_HEADINGS = ("Retrieved", "Available")

_TIMESTAMPED_LINE = re.compile(r"^(\[\d{2}:\d{2}:\d{2}\]) (\[(\w+)\])(.*)$")


def _style_content_line(line: str) -> str:
    """Dim the structural markers of a content line, leaving its text untouched."""
    match = _TIMESTAMPED_LINE.match(line)
    if match:
        timestamp, role_token, role, rest = match.groups()
        return (
            f"{_ANSI_DIM}{timestamp}{_ANSI_RESET} "
            f"{_ANSI_ROLE.get(role.lower(), '')}{role_token}{_ANSI_RESET}{rest}"
        )

    stripped = line.lstrip()
    for marker in ("- ", "• "):
        if stripped.startswith(marker):
            indent = line[:len(line) - len(stripped)]
            return f"{indent}{_ANSI_DIM}{marker.strip()}{_ANSI_RESET} {stripped[len(marker):]}"

    return line


def render_markdown(text: str) -> None:
    """
    Print markdown-shaped memory blocks to the terminal with ANSI styling.

    Styles whole lines by their prefix and never rewraps or rewrites them, so what is
    displayed stays character-identical to what is sent to the model. Falls back to
    plain output when stdout is not a terminal, keeping pipes and logs clean.
    """
    if not sys.stdout.isatty():
        print(text)
        return

    in_preamble = False
    for line in text.splitlines():
        if line.startswith("### "):
            in_preamble = not line[4:].startswith(_CONTENT_HEADINGS)
            print(f"{_ANSI_H3}{line}{_ANSI_RESET}")
        elif line.startswith("## "):
            in_preamble = False
            print(f"{_ANSI_H2}{line}{_ANSI_RESET}")
        elif line.startswith("# "):
            in_preamble = False
            print(f"{_ANSI_H1}{line}{_ANSI_RESET}")
        elif in_preamble:
            print(f"{_ANSI_DIM}{line}{_ANSI_RESET}")
        else:
            print(_style_content_line(line))
