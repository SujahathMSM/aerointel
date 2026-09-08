from scripts.backfill_embeddings import needs_backfill


def test_untagged_row_needs_backfill():
    assert needs_backfill(None, "embeddinggemma") is True


def test_matching_row_is_left_alone():
    assert needs_backfill("embeddinggemma", "embeddinggemma") is False


def test_mismatched_row_needs_backfill():
    assert needs_backfill("old-model", "embeddinggemma") is True
