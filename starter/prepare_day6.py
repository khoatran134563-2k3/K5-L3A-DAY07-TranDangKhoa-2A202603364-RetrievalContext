"""Prepare Day 7 inputs from existing Day 6 files."""

import hashlib
import json
from collections import Counter
from pathlib import Path


DAY6 = Path(
    r"D:\VIN Lab\K5-L3A-DAY06-TranDangKhoa-2A202603364-DataIngestion"
)
DAY7 = Path(__file__).resolve().parents[1]
OUTPUT = DAY7 / "starter" / "data" / "day6"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    chunk_path = DAY6 / "starter/results/chunks/B.jsonl"
    question_path = DAY6 / "benchmark/questions.json"

    raw_chunks = [
        json.loads(line)
        for line in chunk_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw_questions = json.loads(
        question_path.read_text(encoding="utf-8")
    )

    chunks = []
    seen_ids = set()

    for item in raw_chunks:
        chunk_id = item["chunk_id"]

        if chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk ID: {chunk_id}")
        seen_ids.add(chunk_id)

        if not item["text"].strip():
            raise ValueError(f"Empty chunk: {chunk_id}")

        chunks.append({
            "id": chunk_id,
            "source_id": item["document_id"],
            "source": item["source"],
            "page": item["page"],
            "text": item["text"],
            "metadata": item.get("metadata", {}),
        })

    if len(chunks) != 132:
        raise ValueError(
            f"Expected 132 strategy B chunks, found {len(chunks)}"
        )

    questions = []

    for item in raw_questions:
        if item["split"] not in {"dev", "test"}:
            raise ValueError(f"Invalid split: {item['id']}")

        questions.append({
            "id": item["id"],
            "split": item["split"],
            "question": item["question"],
            "answerable": True,
            "expected_answer": item["expected_answer"],
            "gold": [{
                "source_id": item["gold_source_id"],
                "page": item["gold_page"],
                "evidence": item["evidence_quote"],
            }],
            "filter": item.get("filter", {}),
            "review_status": item.get(
                "review_status", "needs_source_review"
            ),
        })

    question_ids = [question["id"] for question in questions]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Duplicate question IDs")

    counts = Counter(question["split"] for question in questions)

    if counts["dev"] < 15 or counts["test"] < 5:
        raise ValueError("Not enough answerable Day 6 questions")

    manifest = {
        "origin": "Day 6 strategy B",
        "document_context": "World Bank WDR 2016 Overview",
        "chunk_count": len(chunks),
        "embedding_model": "text-embedding-3-small",
        "embedding_dimensions": 1536,
        "database_table": "day6_chunks",
        "database_strategy": "B",
        "input_chunks_sha256": sha256(chunk_path),
        "input_questions_sha256": sha256(question_path),
        "question_counts": dict(counts),
        "benchmark_status": (
            "Seed only. Add unanswerable cases, verify evidence "
            "against the original source, and document test reuse "
            "before final evaluation."
        ),
    }

    outputs = {
        "chunks.json": chunks,
        "questions_seed.json": questions,
        "manifest.json": manifest,
    }

    # Refuse to overwrite files from a previous preparation.
    for name in outputs:
        if (OUTPUT / name).exists():
            raise FileExistsError(
                f"Already exists: {OUTPUT / name}"
            )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    for name, data in outputs.items():
        (OUTPUT / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"Chunks: {len(chunks)}")
    print(f"Answerable dev questions: {counts['dev']}")
    print(f"Answerable test questions: {counts['test']}")
    print("Benchmark status: seed, not final.")
    print(f"Saved to: {OUTPUT}")
    print("Preparation OK.")


if __name__ == "__main__":
    main()