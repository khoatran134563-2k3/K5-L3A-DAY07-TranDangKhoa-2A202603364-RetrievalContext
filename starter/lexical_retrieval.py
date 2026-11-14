"""Offline TF-IDF retrieval over the shared Day 7 corpus."""

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def tokenize(text):
    # Shared, deterministic processing for documents and queries.
    return re.findall(r"[a-z0-9]+", text.lower())


class LexicalRetriever:
    def __init__(self):
        corpus_path = ROOT / "starter/data/day6/chunks.json"
        self.chunks = json.loads(
            corpus_path.read_text(encoding="utf-8")
        )

        if not self.chunks:
            raise ValueError("Corpus is empty.")

        ids = [chunk["id"] for chunk in self.chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate chunk IDs.")

        counts = [
            Counter(tokenize(chunk["text"]))
            for chunk in self.chunks
        ]

        document_frequency = Counter()
        for count in counts:
            document_frequency.update(count.keys())

        total_documents = len(self.chunks)

        # Smoothed inverse document frequency.
        self.idf = {
            term: math.log(
                (1 + total_documents) / (1 + frequency)
            ) + 1
            for term, frequency in document_frequency.items()
        }

        self.document_vectors = [
            self.make_vector(count)
            for count in counts
        ]

    def make_vector(self, counts):
        # Sublinear term frequency, then L2 normalization.
        weights = {
            term: (1 + math.log(frequency)) * self.idf[term]
            for term, frequency in counts.items()
            if term in self.idf
        }

        norm = math.sqrt(
            sum(weight * weight for weight in weights.values())
        )

        if norm == 0:
            return {}

        return {
            term: weight / norm
            for term, weight in weights.items()
        }

    def search(self, query, *, filters=None, top_k=3):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a nonempty string.")

        if filters:
            raise ValueError(
                "This initial lexical implementation does not apply filters."
            )

        if not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100.")

        query_vector = self.make_vector(Counter(tokenize(query)))

        if not query_vector:
            return []

        scored = []

        for chunk, vector in zip(
            self.chunks, self.document_vectors
        ):
            score = sum(
                weight * vector.get(term, 0.0)
                for term, weight in query_vector.items()
            )

            # Do not return arbitrary chunks with zero overlap.
            if score > 0:
                scored.append((score, chunk))

        scored.sort(
            key=lambda item: (-item[0], item[1]["id"])
        )

        hits = []

        for rank, (score, chunk) in enumerate(
            scored[:top_k], start=1
        ):
            hits.append({
                **chunk,
                "score": score,
                "rank": rank,
                "method": "tfidf",
            })

        return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    build_started = time.perf_counter()
    retriever = LexicalRetriever()
    build_seconds = time.perf_counter() - build_started

    search_started = time.perf_counter()
    hits = retriever.search(args.query, top_k=args.top_k)
    search_seconds = time.perf_counter() - search_started

    for hit in hits:
        print(
            f"\nRank {hit['rank']} | score={hit['score']:.4f}"
            f" | PDF page={hit['page']}"
        )
        print(f"ID: {hit['id']}")
        print(f"Source: {hit['source']}")
        print(hit["text"])

    output_dir = ROOT / "starter/results/lexical_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "query": args.query,
        "method": "tfidf",
        "top_k": args.top_k,
        "filters": None,
        "configuration": {
            "tokenizer": "lowercase ASCII alphanumeric",
            "term_frequency": "1 + ln(count)",
            "idf": "ln((1 + N) / (1 + df)) + 1",
            "normalization": "L2",
            "stopword_removal": False,
        },
        "index_build_seconds": build_seconds,
        "search_seconds": search_seconds,
        "api_requests": 0,
        "results": hits,
    }

    output_path = output_dir / "latest.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nReturned: {len(hits)} chunks")
    print(f"Index build: {build_seconds:.4f} seconds")
    print(f"Search: {search_seconds:.4f} seconds")
    print("API requests: 0")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()