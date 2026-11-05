"""Mark a stored document revision as superseded (or undo it), or list them.

    python scripts/mark_superseded.py --list          # documents and their revisions
    python scripts/mark_superseded.py 64              # mark document 64 superseded
    python scripts/mark_superseded.py 64 --undo

The system only knows what is stored; it cannot tell when the FAA publishes a
newer revision. When you learn that a stored revision has been replaced, mark
it here: every answer that reads from it then says so in its `sources` note.
Storing the newer revision (same document key) is pointed out automatically.

Requires Postgres (migrated: `alembic upgrade head`).
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.db.base import SessionFactory, engine  # noqa: E402
from app.db.models import Document  # noqa: E402


async def run(document_id: int | None, undo: bool, list_all: bool) -> int:
    try:
        async with SessionFactory() as session:
            if list_all:
                rows = (
                    (await session.execute(select(Document).order_by(Document.id)))
                    .scalars()
                    .all()
                )
                for d in rows:
                    flag = "  SUPERSEDED" if d.superseded else ""
                    print(
                        f"{d.id:>5}  {d.revision or '-'} ({d.revision_date or 'no date'})"
                        f"  key: {d.document_key or '-'}  title: {d.title}{flag}"
                    )
                return 0
            document = await session.get(Document, document_id)
            if document is None:
                print(f"no document {document_id}", file=sys.stderr)
                return 1
            document.superseded = not undo
            await session.commit()
            state = "no longer superseded" if undo else "superseded"
            print(
                f"document {document.id} ({document.revision or document.title}): {state}"
            )
            return 0
    finally:
        await engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description="Mark a document revision superseded.")
    parser.add_argument("document_id", type=int, nargs="?", help="document to mark")
    parser.add_argument("--undo", action="store_true", help="clear the flag instead")
    parser.add_argument("--list", action="store_true", help="list stored documents")
    args = parser.parse_args()
    if args.document_id is None and not args.list:
        parser.error("give a document id, or --list")
    return asyncio.run(run(args.document_id, args.undo, args.list))


if __name__ == "__main__":
    sys.exit(main())
