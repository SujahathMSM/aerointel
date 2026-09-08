"""Prompt registry: prompts are versioned assets on disk, not string literals.

Why this exists. A prompt buried in a Python file has no history you can
point at. You tweak a sentence, answers change, and three weeks later
nothing connects "recall dropped" to "someone reworded rule 3". Here each
prompt is a file named `<name>.<version>.txt`, so a change is a NEW file
reviewed in a diff, old versions stay runnable side by side, and every
answer and eval score can name the exact prompt that produced it.

    from app.prompts import get_prompt

    prompt = get_prompt("grounded_answer")        # newest version
    prompt = get_prompt("grounded_answer", "v1")  # pinned
    prompt.text, prompt.version, prompt.checksum

The checksum matters more than it looks. Version strings are a promise
that the content behind them never changes; a checksum is what keeps that
promise honest. If someone edits v1 in place instead of adding v2, the
checksum moves and last month's eval numbers are revealed as not
comparable to today's — rather than quietly lying.
"""

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent

# <name>.<version>.txt  — version is "v" followed by digits (v1, v2, v10).
_FILENAME = re.compile(r"^(?P<name>[a-z0-9_]+)\.(?P<version>v\d+)\.txt$")


@dataclass(frozen=True)
class Prompt:
    """One resolved prompt: its identity, its text, and proof of content."""

    name: str
    version: str
    text: str
    checksum: str

    @property
    def label(self) -> str:
        """Compact identifier for logs and eval reports: name@version:hash."""
        return f"{self.name}@{self.version}:{self.checksum[:8]}"


def _version_number(version: str) -> int:
    return int(version[1:])


@lru_cache
def _discover() -> dict[str, dict[str, Path]]:
    """Map name -> {version -> path} for every prompt file on disk."""
    found: dict[str, dict[str, Path]] = {}
    for path in PROMPTS_DIR.glob("*.txt"):
        match = _FILENAME.match(path.name)
        if match is None:
            raise ValueError(
                f"prompt file {path.name!r} does not match "
                f"'<name>.<version>.txt' (e.g. grounded_answer.v1.txt)"
            )
        found.setdefault(match["name"], {})[match["version"]] = path
    return found


def list_prompts() -> list[str]:
    """Every prompt name the registry knows about."""
    return sorted(_discover())


def list_versions(name: str) -> list[str]:
    """Known versions of one prompt, oldest first."""
    versions = _discover().get(name)
    if not versions:
        raise KeyError(f"unknown prompt {name!r} (have: {list_prompts()})")
    return sorted(versions, key=_version_number)


def latest_version(name: str) -> str:
    return list_versions(name)[-1]


@lru_cache
def get_prompt(name: str, version: str | None = None) -> Prompt:
    """Load a prompt by name, newest version unless one is pinned.

    Unknown names and versions raise rather than falling back to
    something plausible: a silently wrong prompt is far more expensive to
    notice than a crash at startup.
    """
    versions = _discover().get(name)
    if not versions:
        raise KeyError(f"unknown prompt {name!r} (have: {list_prompts()})")

    resolved = version or latest_version(name)
    path = versions.get(resolved)
    if path is None:
        raise KeyError(
            f"prompt {name!r} has no version {resolved!r} "
            f"(have: {sorted(versions, key=_version_number)})"
        )

    raw = path.read_bytes()
    # Trailing newline is a file convention, not part of the prompt.
    text = raw.decode("utf-8").rstrip("\n")
    return Prompt(
        name=name,
        version=resolved,
        text=text,
        checksum=hashlib.sha256(raw).hexdigest(),
    )
