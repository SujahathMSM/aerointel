"""Inspect what's actually in the AeroIntel database.

Lists every table, describes document_chunks, and prints every row
currently stored (content, metadata, embedding dims - not the raw
768 floats, which are unreadable and useless to look at).

    python scripts/inspect_db.py
"""

import os

import psycopg


def connection_string() -> str:
    return (
        f"host=localhost "
        f"port={os.getenv('POSTGRES_PORT', '5432')} "
        f"dbname={os.environ['POSTGRES_DB']} "
        f"user={os.environ['POSTGRES_USER']} "
        f"password={os.environ['POSTGRES_PASSWORD']}"
    )


def list_tables(cur) -> list[str]:
    cur.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
        """
    )
    return [row[0] for row in cur.fetchall()]


def describe_table(cur, table_name: str) -> list[tuple[str, str, str]]:
    cur.execute(
        """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = %s
        ORDER BY ordinal_position;
        """,
        (table_name,),
    )
    return cur.fetchall()


def fetch_rows(cur):
    cur.execute(
        """
        SELECT id, content, metadata, vector_dims(embedding) AS dims, created_at
        FROM document_chunks
        ORDER BY id;
        """
    )
    return cur.fetchall()


def main() -> None:
    with psycopg.connect(connection_string()) as conn:
        with conn.cursor() as cur:
            tables = list_tables(cur)
            print(f"Tables in database: {tables}")
            print()

            if "document_chunks" not in tables:
                print("document_chunks does not exist. Run sql/001_create_document_chunks.sql.")
                return

            print("document_chunks schema:")
            for column_name, data_type, is_nullable in describe_table(cur, "document_chunks"):
                nullable = "NULL" if is_nullable == "YES" else "NOT NULL"
                print(f"  {column_name:<12} {data_type:<25} {nullable}")
            print()

            rows = fetch_rows(cur)
            print(f"{len(rows)} row(s) in document_chunks:")
            print()

            if not rows:
                print("  (empty)")
                return

            for row_id, content, metadata, dims, created_at in rows:
                print(f"id={row_id}  dims={dims}  created_at={created_at}")
                print(f"  content:  {content}")
                print(f"  metadata: {metadata}")
                print()


if __name__ == "__main__":
    main()