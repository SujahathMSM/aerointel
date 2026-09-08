"""Prompt registry tests.

The load-bearing guarantee here is not "a file can be read" — it is that a
version string always means the same bytes, and that anything unknown
fails loudly instead of silently substituting a plausible prompt.
"""

import pytest

from app.core.config import get_settings
from app.prompts import (
    PROMPTS_DIR,
    get_prompt,
    latest_version,
    list_prompts,
    list_versions,
)
from app.services.generation import PROMPT_NAME, system_prompt


def test_grounded_answer_prompt_exists():
    assert PROMPT_NAME in list_prompts()
    assert list_versions(PROMPT_NAME)  # at least one version on disk


def test_prompt_text_is_the_real_grounding_contract():
    # The whole system's safety rests on these instructions; if a reword
    # ever drops one of them, that is a behavior change, not a typo fix.
    text = get_prompt(PROMPT_NAME, "v1").text
    assert "only the CONTEXT" in text
    assert "[1]" in text  # citation format is specified
    assert "Do not guess" in text
    assert "Do not use outside knowledge" in text


def test_checksum_is_content_addressed():
    # Same version -> same bytes -> same checksum, every time. This is what
    # makes "recall@1 under v1" a comparable number across weeks.
    first = get_prompt(PROMPT_NAME, "v1")
    second = get_prompt(PROMPT_NAME, "v1")
    assert first.checksum == second.checksum
    assert len(first.checksum) == 64  # sha256 hex


def test_label_identifies_name_version_and_content():
    prompt = get_prompt(PROMPT_NAME, "v1")
    assert prompt.label.startswith(f"{PROMPT_NAME}@v1:")
    assert prompt.checksum.startswith(prompt.label.split(":")[-1])


def test_unpinned_resolves_to_latest():
    assert get_prompt(PROMPT_NAME).version == latest_version(PROMPT_NAME)


def test_unknown_prompt_name_fails_loudly():
    with pytest.raises(KeyError, match="unknown prompt"):
        get_prompt("no_such_prompt")


def test_unknown_version_fails_loudly():
    with pytest.raises(KeyError, match="no version"):
        get_prompt(PROMPT_NAME, "v9999")


def test_every_prompt_file_is_nonempty():
    for path in PROMPTS_DIR.glob("*.txt"):
        assert path.read_text(encoding="utf-8").strip(), f"{path.name} is empty"


def test_generation_uses_the_configured_prompt():
    # generation.system_prompt() is what answer_question() actually sends;
    # it must honor the settings pin (None -> latest).
    active = system_prompt()
    expected = get_prompt(PROMPT_NAME, get_settings().prompt_version)
    assert active.version == expected.version
    assert active.text == expected.text


def test_no_prompt_literal_left_in_generation_module():
    # Guards the point of this whole change: if someone reintroduces a
    # SYSTEM_PROMPT string literal, the registry stops being the source of
    # truth and version attribution silently becomes wrong.
    from app.services import generation

    assert not hasattr(generation, "SYSTEM_PROMPT")
