"""The page-based eval for real documents: scoring logic and the golden file.

The real retrieval numbers need Ollama and an ingested manual, so they come
from `python -m evals.doc_eval` on a developer machine. What CI can guarantee
is that the scoring is right and that the golden file is well formed, so a
broken scoreboard cannot hide a retrieval regression.
"""

from evals.doc_eval import (
    answerability_sweep,
    best_ceiling,
    combined_sweep,
    count_facts,
    describe_retrieval,
    is_hit,
    load_golden,
    score_item,
    score_off_topic,
    summarise,
    threshold_sweep,
    unknown_words,
)
from evals.metrics import hit_recall_at_k, hit_reciprocal_rank

SOURCE = "manual"


class Chunk:
    def __init__(self, page, source=SOURCE, content="x"):
        self.page = page
        self.content = content
        self.metadata_ = {"source": source}


def results(*pages_and_distances, source=SOURCE):
    return [(Chunk(page, source), d) for page, d in pages_and_distances]


# ------------------------------------------------------------------ metrics


def test_hit_recall_and_rank():
    hits = [False, False, True, False]
    assert hit_recall_at_k(hits, 2) == 0.0
    assert hit_recall_at_k(hits, 3) == 1.0
    assert hit_reciprocal_rank(hits) == 1 / 3
    assert hit_reciprocal_rank([False, False]) == 0.0
    assert hit_recall_at_k([], 5) == 0.0


# ------------------------------------------------------------------ scoring


def test_a_chunk_is_a_hit_only_on_the_right_page_of_the_right_document():
    assert is_hit(Chunk(394), SOURCE, [394])
    assert not is_hit(Chunk(395), SOURCE, [394])
    assert not is_hit(Chunk(394, source="another manual"), SOURCE, [394])
    assert not is_hit(Chunk(None), SOURCE, [394])


def test_rank_and_distance_of_the_first_right_page():
    row = score_item(
        results((10, 0.20), (11, 0.25), (394, 0.31), (394, 0.40)),
        SOURCE,
        [394],
        max_distance=0.48,
    )
    assert row["rank"] == 3
    assert row["best_hit_distance"] == 0.31
    assert row["recall@1"] == 0.0 and row["recall@3"] == 1.0
    assert row["rr"] == 1 / 3
    assert row["reaches_prompt"] is True


def test_a_right_page_beyond_the_distance_gate_never_reaches_the_prompt():
    row = score_item(results((394, 0.60)), SOURCE, [394], max_distance=0.48)
    assert row["recall@1"] == 1.0  # retrieval found it...
    assert row["reaches_prompt"] is False  # ...but the gate would refuse


def test_a_miss_scores_zero_everywhere():
    row = score_item(results((10, 0.2), (11, 0.3)), SOURCE, [394], 0.48)
    assert row["rank"] is None
    assert row["best_hit_distance"] is None
    assert row["rr"] == 0.0
    assert row["reaches_prompt"] is False


def test_any_of_several_expected_pages_counts():
    row = score_item(results((395, 0.2)), SOURCE, [394, 395], 0.48)
    assert row["rank"] == 1


def test_context_hit_sees_the_widened_context_not_just_the_hits():
    hits = results((394, 0.3))
    # the hit is on 7-39, the answer continues on the next page
    widened = [*hits, (Chunk(395), 0.3)]

    on_first_page = score_item(hits, SOURCE, [395], 0.48, context=hits)
    after_widening = score_item(hits, SOURCE, [395], 0.48, context=widened)

    assert on_first_page["rank"] is None
    assert on_first_page["context_hit"] is False
    assert after_widening["context_hit"] is True


# ------------------------------------------------------------------- facts


def ctx(*texts):
    return [(Chunk(394, content=t), 0.3) for t in texts]


def test_facts_are_counted_across_all_the_chunks_the_model_reads():
    context = ctx("Inspect for broken wires.", "Pass a cloth over the area.")
    assert count_facts(context, ["broken wires", "a cloth over the area"]) == 2
    assert count_facts(context, ["broken wires", "magnifying glass"]) == 1


def test_fact_matching_ignores_case_and_spacing():
    context = ctx("A swaged-on   FITTING can be\nan eye or fork.")
    assert count_facts(context, ["swaged-on fitting can be an eye"]) == 1


def test_the_sentence_that_answers_the_question_can_be_missing_from_the_right_page():
    # the hit stops at "c."; the answer ("pass a cloth") is in the next chunk
    hit = (Chunk(394, content="a. Inspect at each annual. b. Define the area. c."), 0.3)
    follower = (Chunk(394, content="d. Pass a cloth over the area to snag wires."), 0.3)
    facts = ["pass a cloth over the area"]

    plain = score_item([hit], SOURCE, [394], 0.48, [hit], facts)
    widened = score_item([hit], SOURCE, [394], 0.48, [hit, follower], facts)

    assert plain["context_hit"] is True  # right page, yet...
    assert plain["facts_found"] == 0  # ...the answer is not in front of the model
    assert widened["facts_found"] == 1


def test_no_facts_listed_means_no_fact_columns():
    row = score_item(results((394, 0.3)), SOURCE, [394], 0.48, results((394, 0.3)))
    assert "facts_total" not in row


def test_fact_coverage_is_the_mean_share_and_all_facts_the_share_complete():
    rows = [
        score_item(results((5, 0.2)), SOURCE, [5], 0.48, ctx("a b"), ["a", "b"]),
        score_item(results((5, 0.2)), SOURCE, [5], 0.48, ctx("a"), ["a", "b"]),
    ]
    summary = summarise(rows, [])
    assert summary["fact_coverage"] == (1.0 + 0.5) / 2
    assert summary["all_facts"] == 0.5


def test_off_topic_is_refused_when_nothing_is_close():
    assert score_off_topic(results((1, 0.82)), 0.48)["refused"] is True
    assert score_off_topic(results((1, 0.40)), 0.48)["refused"] is False
    assert score_off_topic([], 0.48)["refused"] is True


def test_summary_means():
    rows = [
        score_item(results((5, 0.2)), SOURCE, [5], 0.48),
        score_item(results((6, 0.2), (5, 0.3)), SOURCE, [5], 0.48),
        score_item(results((6, 0.2)), SOURCE, [5], 0.48),
    ]
    off = [score_off_topic(results((1, 0.9)), 0.48)]
    summary = summarise(rows, off)
    assert summary["questions"] == 3
    assert summary["recall@1"] == 1 / 3
    assert summary["recall@3"] == 2 / 3
    assert summary["mrr"] == (1 + 0.5 + 0) / 3
    assert summary["off_topic_refused"] == 1.0
    assert "context_hit" not in summary  # only when widening was scored


# ------------------------------------------------------------- the threshold


def test_sweep_and_best_ceiling():
    rows = [
        {"best_hit_distance": 0.30, "top_distance": 0.30},
        {"best_hit_distance": 0.40, "top_distance": 0.40},
        {"best_hit_distance": 0.52, "top_distance": 0.52},
    ]
    off = [{"top_distance": 0.70}, {"top_distance": 0.62}]
    sweep = threshold_sweep(rows, off)

    at = {s["max_distance"]: s for s in sweep}
    assert at[0.30]["reaches_prompt"] == 1 / 3
    assert at[0.52]["reaches_prompt"] == 1.0
    assert at[0.52]["off_topic_refused"] == 1.0
    assert at[0.64]["off_topic_refused"] == 0.5  # the 0.62 question now gets through

    best = best_ceiling(sweep)
    assert best["reaches_prompt"] == 1.0
    assert best["off_topic_refused"] == 1.0
    # every ceiling from 0.52 (all right pages in) to 0.60 (0.62 still refused)
    # works; the middle of that range leaves the most room on both sides
    assert (best["low"], best["high"]) == (0.52, 0.60)
    assert best["max_distance"] == 0.56


def test_no_safe_ceiling_when_off_topic_is_closer_than_right_answers():
    rows = [{"best_hit_distance": 0.60, "top_distance": 0.60}]
    off = [{"top_distance": 0.31}]
    # a ceiling that lets the right page in also lets the off-topic one in
    assert best_ceiling(threshold_sweep(rows, off)) is None


# ------------------------------------------------------------- the golden file


def test_the_golden_file_is_well_formed():
    golden = load_golden()
    assert golden["source"]
    items = golden["items"]
    assert len(items) >= 20

    ids = [item["id"] for item in items] + [o["id"] for o in golden["off_topic"]]
    assert len(ids) == len(set(ids)), "question ids must be unique"

    questions = [item["question"] for item in items]
    assert len(questions) == len(set(questions))
    for item in items:
        assert item["question"].strip().endswith("?")
        assert item["pages"] and all(
            isinstance(p, int) and p > 0 for p in item["pages"]
        )
        assert len(item["labels"]) == len(item["pages"])
        assert item["facts"] and all(f.strip() for f in item["facts"])

    # generic off-topic questions plus aviation ones the manual does not cover
    assert len(golden["off_topic"]) >= 10
    kinds = [o["kind"] for o in golden["off_topic"]]
    assert set(kinds) <= {"generic", "outside", "near_topic"}
    assert kinds.count("near_topic") >= 15
    for o in golden["off_topic"]:
        assert o["question"].strip()
        if o["kind"] == "near_topic":
            assert o["manual_says"].strip(), o["id"]


# ------------------------------------------------- answerability + --show


def test_unknown_words_depend_on_how_many_chunks_are_required():
    counts = {"airbus": 0, "helicopter": 1, "cable": 40}
    assert unknown_words(counts, 1) == ["airbus"]
    assert unknown_words(counts, 2) == ["airbus", "helicopter"]


def test_answerability_sweep_counts_wrong_refusals_and_catches():
    valid = [{"cable": 9}, {"judging": 0, "fabric": 4}]
    off = [
        ({"airbus": 0, "a320": 0}, False),  # the distance gate let it through
        ({"boeing": 0}, False),
        ({"capital": 0}, True),  # the gate already refused it: not counted
        ({"carbon": 50, "propeller": 120}, False),  # every word occurs: not caught
    ]
    rows = {
        (r["min_chunks"], r["min_unknown"]): r for r in answerability_sweep(valid, off)
    }
    strict = rows[(1, 1)]
    assert (strict["valid_refused"], strict["valid_total"]) == (1, 2)
    assert (strict["caught"], strict["gate_missed"]) == (2, 3)
    relaxed = rows[(1, 2)]
    assert relaxed["valid_refused"] == 0
    assert relaxed["caught"] == 1  # only the question with two unknown words


def _row(distance, **counts):
    return {
        "best_hit_distance": distance,
        "top_distance": distance,
        "term_counts": counts or {"cable": 5},
    }


def test_the_vocabulary_rule_makes_a_looser_ceiling_safe():
    right = [_row(0.30), _row(0.479)]
    off = [
        _row(0.45, airbus=0),  # foreign vocabulary, nearer than the worst right answer
        _row(0.80, cake=0),
    ]
    plain = best_ceiling(threshold_sweep(right, off, low=0.40, high=0.62))
    assert plain["reaches_prompt"] == 0.5  # to refuse the Airbus question, lose one

    combined = best_ceiling(combined_sweep(right, off, 1, 1))
    assert combined["reaches_prompt"] == 1.0
    assert combined["low"] == 0.48 and combined["high"] == 0.62


def test_the_vocabulary_rule_also_costs_right_questions_that_use_a_foreign_word():
    right = [_row(0.30), _row(0.40, judging=0)]
    off = [_row(0.80, cake=0)]
    best = best_ceiling(combined_sweep(right, off, 1, 1))
    assert best["reaches_prompt"] == 0.5
    relaxed = best_ceiling(combined_sweep(right, off, 1, 2))
    assert relaxed["reaches_prompt"] == 1.0


class ShownChunk(Chunk):
    def __init__(self, cid, page, content, label=None, index=0):
        super().__init__(page, content=content)
        self.id = cid
        self.page_label = label
        self.chunk_index = index


def test_show_marks_the_right_page_and_says_where_each_fact_is():
    found = [
        (
            ShownChunk(
                11, 349, "Noble metal coatings... clean with magnesium.", "6-42"
            ),
            0.456,
        ),
        (ShownChunk(12, 100, "Unrelated rivet text."), 0.50),
    ]
    text = describe_retrieval(
        "titanium",
        "Why does titanium change colour?",
        found,
        SOURCE,
        [349],
        ["blue oxide coating", "clean with magnesium"],
    )
    assert "expected PDF page(s): [349]" in text
    assert "#1 chunk 11 PDF p.349 (6-42)" in text and "<- right page" in text
    assert text.count("<- right page") == 1
    assert "fact 'blue oxide coating': NOT in any shown chunk" in text
    assert "fact 'clean with magnesium': in result 1" in text


# ------------------------------------------- served retrieval and reading limit


def test_the_gate_is_judged_on_the_closest_served_chunk_not_the_first():
    # hybrid order: the keyword-ranked first result is further than the third
    served = results((1, 0.55), (2, 0.60), (3, 0.47))
    off = score_off_topic(served, 0.48)
    assert off["top_distance"] == 0.55
    assert off["gate_distance"] == 0.47
    assert off["refused"] is False  # the API would answer it


def test_a_right_page_past_the_gate_reaches_the_model_only_with_a_reading_limit():
    # the weld question as served live: passes on page 463, answer on 464 at 0.495
    served = results((463, 0.422), (464, 0.495))
    plain = score_item(served, SOURCE, [464], 0.48, served=served)
    widened = score_item(
        served, SOURCE, [464], 0.48, served=served, context_max_distance=0.52
    )
    assert plain["gate_distance"] == 0.422
    assert plain["served_hit_distance"] == 0.495
    assert plain["reaches_prompt"] is False
    assert widened["reaches_prompt"] is True


def test_a_reading_limit_does_not_help_a_question_the_gate_refuses():
    served = results((464, 0.495))
    row = score_item(
        served, SOURCE, [464], 0.48, served=served, context_max_distance=0.60
    )
    assert row["reaches_prompt"] is False


def test_rank_comes_from_the_deep_ranking_and_reach_from_what_is_served():
    deep = results(
        (10, 0.30), (11, 0.31), (12, 0.32), (13, 0.33), (14, 0.34), (5, 0.35)
    )
    served = deep[:5]  # the right page is 6th: never served
    row = score_item(deep, SOURCE, [5], 0.48, served=served)
    assert row["rank"] == 6
    assert row["best_hit_distance"] == 0.35
    assert row["served_hit_distance"] is None
    assert row["reaches_prompt"] is False


def test_sweep_with_a_reading_limit():
    rows = [
        {"gate_distance": 0.42, "served_hit_distance": 0.495},  # weld
        {"gate_distance": 0.40, "served_hit_distance": 0.40},
    ]
    off = [{"gate_distance": 0.465}]
    at = {s["max_distance"]: s for s in threshold_sweep(rows, off)}
    widened = {
        s["max_distance"]: s
        for s in threshold_sweep(rows, off, context_max_distance=0.52)
    }
    assert at[0.44]["reaches_prompt"] == 0.5  # weld's answer is never read
    assert widened[0.44]["reaches_prompt"] == 1.0  # read past the gate
    assert widened[0.44]["off_topic_refused"] == 1.0  # the gate is unchanged
    assert widened[0.40]["reaches_prompt"] == 0.5  # weld fails the gate at 0.40


def test_sweep_still_reads_result_files_saved_before_this_change():
    old_rows = [{"best_hit_distance": 0.30, "top_distance": 0.30}]
    old_off = [{"top_distance": 0.70}]
    at = {s["max_distance"]: s for s in threshold_sweep(old_rows, old_off)}
    assert at[0.30]["reaches_prompt"] == 1.0
    assert at[0.30]["off_topic_refused"] == 1.0


def test_show_marks_what_the_model_reads():
    served = [
        (ShownChunk(1, 463, "Remove all flux after welding."), 0.422),
        (ShownChunk(2, 464, "Tanks must not be welded unless purged."), 0.495),
    ]
    text = describe_retrieval(
        "weld", "q?", served, SOURCE, [464], None, read=served[:1]
    )
    lines = [line for line in text.splitlines() if line.startswith("  #")]
    assert lines[0].endswith("[read]")
    assert lines[1].endswith("<- right page  [not read]")


# ------------------------------------------------ near-topic questions


def test_near_topic_questions_are_kept_out_of_the_gate_sweep():
    from evals.doc_eval import near_topic_summary, split_off_topic

    off = [
        {"id": "a", "kind": "outside", "refused": True, "top_distance": 0.53},
        {"id": "b", "refused": False, "top_distance": 0.45},  # old file: no kind
        {"id": "c", "kind": "near_topic", "refused": False, "top_distance": 0.36},
        {"id": "d", "kind": "near_topic", "refused": False, "top_distance": 0.44},
        {"id": "e", "kind": "near_topic", "refused": True, "top_distance": 0.55},
    ]
    gate, near = split_off_topic(off)
    assert [r["id"] for r in gate] == ["a", "b"]
    assert [r["id"] for r in near] == ["c", "d", "e"]
    assert near_topic_summary(near) == {
        "total": 3,
        "gate_refused": 1,
        "passed_min_distance": 0.36,
        "passed_max_distance": 0.44,
    }
    assert near_topic_summary([]) is None


def test_a_near_topic_question_does_not_pull_the_suggested_ceiling_down():
    from evals.doc_eval import best_ceiling, split_off_topic, threshold_sweep

    rows = [{"best_hit_distance": 0.40, "top_distance": 0.40}]
    off = [
        {"id": "far", "kind": "outside", "refused": True, "top_distance": 0.60},
        {"id": "near", "kind": "near_topic", "refused": False, "top_distance": 0.36},
    ]
    gate, _near = split_off_topic(off)
    best = best_ceiling(threshold_sweep(rows, gate))
    assert best is not None and best["reaches_prompt"] == 1.0
    # With the near-topic question in the sweep, no ceiling lets the right
    # question through and refuses everything.
    mixed = best_ceiling(threshold_sweep(rows, off))
    assert mixed is None or mixed["reaches_prompt"] < 1.0
