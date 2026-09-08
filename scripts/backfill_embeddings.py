"""Backfill embedding_model, or re-embed everything under a new model.

Different embedding models produce incompatible vectors, so every row
carries embedding_model — which model actually produced its `embedding`
column. This script is what keeps that column honest, in two situations:

  1. Backfill (no flags): rows written before embedding-version tracking
     existed have embedding_model = NULL. Stamp them with the model that
     was configured when they were embedded (default: settings' current
     embedding_model — correct only if the model hasn't changed since;
     for anything more precise, know your own history).

  2. Model upgrade (--model): point at a new Ollama model and re-embed
     every row under it. Rows keep their OLD embedding_model tag until
     THIS script actually overwrites them, so a crash partway through
     leaves a mix of old- and new-tagged rows — visible, not silently
     wrong. Re-run to finish; already-matching rows are skipped. Only
     flip settings.embedding_model to the new model after a clean run.

    python scripts/backfill_embeddings.py                    # tag NULL rows with the current model
    python scripts/backfill_embeddings.py --model newmodel   # re-embed everyone under newmodel
    python scripts/backfill_embeddings.py --dry-run          # show what would change, write nothing

Requires: Postgres up, Ollama running with the target model pulled.

Caveat: embedding_dimensions (the Vector column's fixed width) does not
change here. A target model whose output dimensionality differs from the
current column width will fail loudly on the first embed call rather than
corrupt data — but actually widening the column is a separate schema
migration, out of scope for this script.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Run as `python scripts/backfill_embeddings.py` from the repo root: that puts
# scripts/ on sys.path, not the root, so `app` would not import. Add the repo
# root explicitly rather than forcing a `python -m` invocation on the caller.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.db.base import SessionFactory  # noqa: E402
from app.db.models import DocumentChunk  # noqa: E402
from app.llm.ollama import OllamaClient  # noqa: E402


def needs_backfill(current_model: str | None, target_model: str) -> bool:
    """True if a row's embedding was NOT made by target_model.

    Covers both cases this script handles: never-tagged rows
    (current_model is None, from before embedding versioning existed)
    and rows tagged with some other model (an in-progress or reverted
    upgrade).
    """
    return current_model != target_model


async def run(target_model: str, dry_run: bool, batch_size: int = 16) -> int:
    async with SessionFactory() as session:
        rows = (await session.execute(select(DocumentChunk))).scalars().all()

    stale = [row for row in rows if needs_backfill(row.embedding_model, target_model)]

    if not stale:
        print(
            f"Nothing to backfill: every row already tagged "
            f"embedding_model={target_model!r}."
        )
        return 0

    print(f"{len(stale)} of {len(rows)} row(s) need embedding_model={target_model!r}.")
    for row in stale:
        current = row.embedding_model or "(untagged)"
        print(f"  id={row.id:<6} currently {current!r} -> {target_model!r}")

    if dry_run:
        print("\n--dry-run: no changes written.")
        return 0

    # Direct the client at target_model specifically — it may differ from
    # whatever is currently configured (that's the whole point of --model).
    ollama = OllamaClient(
        settings=get_settings().model_copy(update={"embedding_model": target_model})
    )
    updated = 0
    try:
        async with SessionFactory() as session:
            for i in range(0, len(stale), batch_size):
                batch = stale[i : i + batch_size]
                embeddings = await ollama.embed([row.content for row in batch])
                for row, embedding in zip(batch, embeddings):
                    db_row = await session.get(DocumentChunk, row.id)
                    if db_row is None:
                        continue  # deleted since the scan above; skip, not fatal
                    db_row.embedding = embedding
                    db_row.embedding_model = target_model
                    updated += 1
                await session.commit()
                print(f"  committed {min(i + batch_size, len(stale))}/{len(stale)}")
    finally:
        await ollama.close()

    print(
        f"\nDone: {updated} row(s) re-embedded under embedding_model={target_model!r}."
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        default=None,
        help="target embedding model (default: the currently configured embedding_model)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="show what would change, write nothing"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="rows embedded per Ollama call (default: 16)",
    )
    args = parser.parse_args()

    target = args.model or get_settings().embedding_model
    raise SystemExit(asyncio.run(run(target, args.dry_run, args.batch_size)))
