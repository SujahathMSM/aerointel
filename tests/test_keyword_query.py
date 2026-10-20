"""How the keyword half of hybrid search reads a question (no database)."""

import pytest

from app.core.config import get_settings
from app.services.hybrid import keyword_query_text


def test_all_mode_passes_the_question_through():
    q = "What is the torque for a 1/4 inch bolt?"
    assert keyword_query_text(q, "all") == q


def test_any_mode_joins_distinctive_words_with_or():
    # websearch_to_tsquery reads "or" as the OR operator.
    text = keyword_query_text("What is the torque for a 1/4 inch bolt?", "any")
    assert text == "torque or inch or bolt"


def test_any_mode_keeps_letter_digit_words():
    text = keyword_query_text("Is the G1000 wiring covered?", "any")
    assert text is not None
    assert "g1000" in text.split(" or ")


def test_any_mode_with_no_distinctive_words_searches_nothing():
    assert keyword_query_text("what is it?", "any") is None


def test_unknown_mode_is_an_error():
    with pytest.raises(ValueError):
        keyword_query_text("bolt", "some")


def test_default_is_any_as_measured():
    assert get_settings().keyword_query == "any"
