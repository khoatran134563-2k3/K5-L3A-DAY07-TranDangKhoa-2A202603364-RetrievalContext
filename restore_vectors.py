"""Restore strategy B explicitly from bundled vectors. No embedding API."""

import argparse
import json
import os

from run_portable import ROOT, prepare_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace-b", action="store_true")
    args = parser.parse_args()

    if not args.replace_b:
        raise SystemExit(
            "Use --replace-b to explicitly replace strategy B in day6_chunks."
        )

    prepare_runtime()
    from src.models import Chunk
    from src.store import PgVectorStore

    records = [
        json.loads(line)
        for line in (ROOT / "runtime/B.jsonl")
        .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    corpus = json.loads(
        (ROOT / "starter/data/day6/chunks.json")
        .read_text(encoding="utf-8")
    )
    by_id = {chunk["id"]: chunk for chunk in corpus}

    if len(records) != 132 or len(by_id) != 132:
        raise ValueError("Expected 132 unique corpus chunks.")

    chunks = []
    vectors = []
    seen = set()

    for record in records:
        cid = record["chunk_id"]
        if cid in seen:
            raise ValueError("Duplicate vector chunk ID.")
        seen.add(cid)

        if (
            record["embedding_model"] != "text-embedding-3-small"
            or record["embedding_dimensions"] != 1536
        ):
            raise ValueError("Unexpected embedding configuration.")

        expected = by_id[cid]
        if (
            record["document_id"] != expected["source_id"]
            or record["source"] != expected["source"]
            or record["page"] != expected["page"]
            or record["text"] != expected["text"]
        ):
            raise ValueError(f"Corpus mismatch: {cid}")

        metadata = dict(record["metadata"])
        metadata["embedding_model"] = record["embedding_model"]

        chunks.append(Chunk(
            chunk_id=cid,
            document_id=record["document_id"],
            source=record["source"],
            page=record["page"],
            text=record["text"],
            metadata=metadata,
        ))
        vectors.append(record["embedding"])

    store = PgVectorStore(os.environ["DATABASE_URL"], 1536)
    store.create_schema()
    store.upsert("B", chunks, vectors)
    print("Restored strategy B:", store.snapshot("B"))
    print("No embedding API calls.")


if __name__ == "__main__":
    main()