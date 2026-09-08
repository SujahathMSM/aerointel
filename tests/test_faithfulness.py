"""Faithfulness scoring tests — all pure, no database, no models.

The deterministic half (citation validity) is the half that can be gated
on, so it gets the harder cases: multi-citations, out-of-range blocks,
and the boundary conditions where an off-by-one would silently pass.
"""

from evals.faithfulness import (
    build_judge_input,
    citation_validity,
    cited_indices,
    has_hallucinated_citation,
    mean_or_none,
    parse_judge_verdict,
)

# --- citation parsing -----------------------------------------------------


def test_single_citation():
    assert cited_indices("Check the servo valve [1].") == {1}


def test_grouped_citation():
    assert cited_indices("Both apply [2, 3].") == {2, 3}


def test_grouped_citation_without_spaces():
    assert cited_indices("Both apply [2,3].") == {2, 3}


def test_multiple_separate_citations():
    assert cited_indices("First [1]. Second [3]. Again [1].") == {1, 3}


def test_no_citations():
    assert cited_indices("The manual says to inspect the filter.") == set()


# --- citation validity ----------------------------------------------------


def test_all_citations_valid():
    assert citation_validity("Per [1] and [2].", n_context_blocks=3) == 1.0


def test_uncited_answer_scores_zero():
    # Citations are part of the grounding contract; omitting them is a
    # failure of that contract, not a neutral result.
    assert citation_validity("Inspect the filter.", n_context_blocks=3) == 0.0


def test_hallucinated_citation_drags_score_down():
    # [4] does not exist when only 3 blocks were supplied: 1 of 2 valid.
    assert citation_validity("Per [1] and [4].", n_context_blocks=3) == 0.5


def test_all_citations_hallucinated():
    assert citation_validity("Per [7].", n_context_blocks=3) == 0.0


def test_last_block_is_in_range():
    # Off-by-one guard: block N is valid when N blocks were supplied.
    assert citation_validity("Per [3].", n_context_blocks=3) == 1.0


def test_zero_index_is_out_of_range():
    # Blocks are numbered from 1; [0] is not a real source.
    assert citation_validity("Per [0].", n_context_blocks=3) == 0.0


def test_has_hallucinated_citation_flags_only_out_of_range():
    assert has_hallucinated_citation("Per [4].", 3) is True
    assert has_hallucinated_citation("Per [1, 2].", 3) is False
    assert has_hallucinated_citation("No citation here.", 3) is False


# --- judge parsing --------------------------------------------------------


def test_parse_supported():
    assert parse_judge_verdict("VERDICT: SUPPORTED\nREASON: all figures present") == 1.0


def test_parse_unsupported():
    assert parse_judge_verdict("VERDICT: UNSUPPORTED\nREASON: invented 200 bar") == 0.0


def test_parse_is_case_insensitive():
    assert parse_judge_verdict("verdict: supported\nreason: fine") == 1.0


def test_parse_tolerates_surrounding_text():
    reply = "Here is my grade.\nVERDICT: UNSUPPORTED\nREASON: added a limit\nThanks!"
    assert parse_judge_verdict(reply) == 0.0


def test_unparseable_judge_is_none_not_zero():
    # A broken judge must not be recorded as a hallucinating answer, or the
    # metric measures the judge's failures instead of the model's.
    assert parse_judge_verdict("I'm not sure how to grade this.") is None
    assert parse_judge_verdict("") is None


# --- aggregation ----------------------------------------------------------


def test_mean_ignores_unparseable():
    assert mean_or_none([1.0, 0.0, None, 1.0]) == 2 / 3


def test_mean_all_unparseable_is_none():
    assert mean_or_none([None, None]) is None


def test_mean_empty_is_none():
    assert mean_or_none([]) is None


# --- judge input ----------------------------------------------------------


def test_judge_input_carries_all_three_parts():
    payload = build_judge_input("q?", "[1] block text", "answer [1]")
    assert "q?" in payload
    assert "[1] block text" in payload
    assert "answer [1]" in payload
