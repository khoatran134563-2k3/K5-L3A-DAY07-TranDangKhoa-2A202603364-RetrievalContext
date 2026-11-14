"""Dense retrieval using the existing Day 6 pgvector database."""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DAY6_STARTER = Path(
    r"D:\VIN Lab\K5-L3A-DAY06-TranDangKhoa-2A202603364-DataIngestion\starter"
)

MODEL = "text-embedding-3-small"
DIMENSIONS = 1536

# Reuse only the database adapter. Do not run Day 6 ingestion.
sys.path.insert(0, str(DAY6_STARTER))
from src.store import PgVectorStore


class DenseRetriever:
    def __init__(self):
        load_dotenv(DAY6_STARTER / ".env", override=False)

        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL is missing.")

        self.store = PgVectorStore(database_url, DIMENSIONS)
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()

        self.cache_dir = ROOT / "starter/results/query_embedding_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        corpus_path = ROOT / "starter/data/day6/chunks.json"
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        self.corpus_by_id = {chunk["id"]: chunk for chunk in corpus}

        if len(self.corpus_by_id) != len(corpus):
            raise ValueError("Duplicate corpus chunk IDs.")

        self.last_usage = {}

    @staticmethod
    def validate_vector(vector):
        if not isinstance(vector, list) or len(vector) != DIMENSIONS:
            raise ValueError("Unexpected embedding dimensions.")

        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in vector
        ):
            raise ValueError("Invalid embedding values.")

        if not any(value != 0 for value in vector):
            raise ValueError("Embedding cannot be a zero vector.")

    def embed_query(self, query):
        identity = json.dumps(
            [MODEL, DIMENSIONS, query],
            ensure_ascii=False,
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        cache_path = self.cache_dir / f"{digest}.json"

        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            vector = cached["embedding"]
            self.validate_vector(vector)

            self.last_usage = {
                "embedding_model": MODEL,
                "cache_hit": True,
                "api_requests": 0,
                "total_tokens": 0,
            }
            return vector

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is missing.")

        payload = {
            "model": MODEL,
            "dimensions": DIMENSIONS,
            "encoding_format": "float",
            "input": query,
        }

        request = Request(
            "https://api.openai.com/v1/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=60) as response:
                result = json.load(response)
        except HTTPError as error:
            raise RuntimeError(
                f"Embedding API HTTP {error.code}; no mock fallback."
            ) from None
        except (URLError, TimeoutError, ConnectionError):
            raise RuntimeError(
                "Embedding connection failed; no mock fallback."
            ) from None

        vector = result["data"][0]["embedding"]
        self.validate_vector(vector)

        self.last_usage = {
            "embedding_model": result.get("model", MODEL),
            "cache_hit": False,
            "api_requests": 1,
            "total_tokens": result["usage"]["total_tokens"],
        }

        cache_path.write_text(
            json.dumps({
                "model": MODEL,
                "dimensions": DIMENSIONS,
                "embedding": vector,
                "original_usage": self.last_usage,
            }),
            encoding="utf-8",
        )

        return vector

    def search(self, query, *, filters=None, top_k=3):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a nonempty string.")

        if filters:
            raise ValueError(
                "This initial dense implementation does not apply filters."
            )

        if not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100.")

        vector = self.embed_query(query)

        # Strategy B is the shared corpus for all Day 7 methods.
        rows = self.store.search(
            strategy="B",
            query_vector=vector,
            top_k=top_k,
        )

        hits = []

        for rank, row in enumerate(rows, start=1):
            chunk_id = row["chunk_id"]
            expected = self.corpus_by_id.get(chunk_id)

            if expected is None:
                raise ValueError(
                    f"Database chunk absent from Day 7 corpus: {chunk_id}"
                )

            actual_identity = (
                row["document_id"],
                row["source"],
                row["page"],
                row["text"],
            )
            expected_identity = (
                expected["source_id"],
                expected["source"],
                expected["page"],
                expected["text"],
            )

            if actual_identity != expected_identity:
                raise ValueError(
                    f"Database/corpus content mismatch: {chunk_id}"
                )

            hits.append({
                "id": chunk_id,
                "source_id": row["document_id"],
                "source": row["source"],
                "page": row["page"],
                "text": row["text"],
                "metadata": row["metadata"],
                "score": row["score"],
                "rank": rank,
                "method": "dense",
            })

        return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    retriever = DenseRetriever()

    started = time.perf_counter()
    hits = retriever.search(args.query, top_k=args.top_k)
    elapsed = time.perf_counter() - started

    for hit in hits:
        print(
            f"\nRank {hit['rank']} | score={hit['score']:.4f}"
            f" | PDF page={hit['page']}"
        )
        print(f"ID: {hit['id']}")
        print(f"Source: {hit['source']}")
        print(hit["text"])

    output_dir = ROOT / "starter/results/dense_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "query": args.query,
        "method": "dense",
        "strategy": "B",
        "top_k": args.top_k,
        "filters": None,
        "elapsed_seconds": elapsed,
        "embedding_usage": retriever.last_usage,
        "results": hits,
    }

    output_path = output_dir / "latest.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nReturned: {len(hits)} chunks")
    print(f"Elapsed: {elapsed:.3f} seconds")
    print(f"Embedding usage: {retriever.last_usage}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()