"""Hybrid retrieval using Reciprocal Rank Fusion."""

import argparse
import json
import time
from pathlib import Path

from dense_retrieval import DenseRetriever
from lexical_retrieval import LexicalRetriever


ROOT = Path(__file__).resolve().parents[1]


class HybridRetriever:
    def __init__(self, candidate_k=10, rrf_k=60):
        if not isinstance(candidate_k, int) or not 1 <= candidate_k <= 100:
            raise ValueError("candidate_k must be between 1 and 100.")

        if not isinstance(rrf_k, int) or rrf_k < 1:
            raise ValueError("rrf_k must be a positive integer.")

        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

        self.dense = DenseRetriever()
        self.lexical = LexicalRetriever()

        self.last_trace = {}

    def search(self, query, *, filters=None, top_k=3):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a nonempty string.")

        if filters:
            raise ValueError(
                "This initial hybrid implementation does not apply filters."
            )

        if not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100.")

        if top_k > self.candidate_k:
            raise ValueError(
                "top_k must not exceed candidate_k."
            )

        started = time.perf_counter()

        dense_started = time.perf_counter()
        dense_hits = self.dense.search(
            query,
            top_k=self.candidate_k,
        )
        dense_seconds = time.perf_counter() - dense_started

        lexical_started = time.perf_counter()
        lexical_hits = self.lexical.search(
            query,
            top_k=self.candidate_k,
        )
        lexical_seconds = time.perf_counter() - lexical_started

        fused = {}

        for method, hits in (
            ("dense", dense_hits),
            ("tfidf", lexical_hits),
        ):
            for rank, hit in enumerate(hits, start=1):
                chunk_id = hit["id"]

                if chunk_id not in fused:
                    fused[chunk_id] = {
                        "id": chunk_id,
                        "source_id": hit["source_id"],
                        "source": hit["source"],
                        "page": hit["page"],
                        "text": hit["text"],
                        "metadata": hit["metadata"],
                        "score": 0.0,
                        "component_ranks": {},
                        "component_scores": {},
                    }
                else:
                    existing = fused[chunk_id]
                    for field in (
                        "source_id", "source", "page", "text"
                    ):
                        if existing[field] != hit[field]:
                            raise ValueError(
                                f"Inconsistent chunk: {chunk_id}"
                            )

                item = fused[chunk_id]

                # Equal-weight Reciprocal Rank Fusion.
                item["score"] += 1.0 / (self.rrf_k + rank)
                item["component_ranks"][method] = rank
                item["component_scores"][method] = hit["score"]

        ranked = sorted(
            fused.values(),
            key=lambda item: (-item["score"], item["id"]),
        )

        results = [
            {
                **item,
                "rank": rank,
                "method": "hybrid_rrf",
            }
            for rank, item in enumerate(
                ranked[:top_k], start=1
            )
        ]

        self.last_trace = {
            "query": query,
            "candidate_k_per_method": self.candidate_k,
            "rrf_k": self.rrf_k,
            "weights": {"dense": 1.0, "tfidf": 1.0},
            "dense_seconds": dense_seconds,
            "lexical_seconds": lexical_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "embedding_usage": dict(self.dense.last_usage),
            "dense_candidates": dense_hits,
            "lexical_candidates": lexical_hits,
            "unique_candidates": len(fused),
        }

        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)
    args = parser.parse_args()

    build_started = time.perf_counter()
    retriever = HybridRetriever(
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
    )
    initialization_seconds = (
        time.perf_counter() - build_started
    )

    hits = retriever.search(
        args.query,
        top_k=args.top_k,
    )

    for hit in hits:
        print(
            f"\nRank {hit['rank']}"
            f" | RRF score={hit['score']:.6f}"
            f" | PDF page={hit['page']}"
        )
        print(f"Component ranks: {hit['component_ranks']}")
        print(f"ID: {hit['id']}")
        print(f"Source: {hit['source']}")
        print(hit["text"])

    output_dir = ROOT / "starter/results/hybrid_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "query": args.query,
        "method": "hybrid_rrf",
        "top_k": args.top_k,
        "filters": None,
        "initialization_seconds": initialization_seconds,
        "trace": retriever.last_trace,
        "results": hits,
    }

    output_path = output_dir / "latest.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nReturned: {len(hits)} chunks")
    print(
        "Search elapsed: "
        f"{retriever.last_trace['elapsed_seconds']:.4f} seconds"
    )
    print(
        f"Embedding usage: {retriever.last_trace['embedding_usage']}"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()