"""Check dev gold evidence against the shared corpus without retrieval."""

import json
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "starter/data/day6"
OUTPUT = ROOT / "starter/results/benchmark_audit"


def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    chunks = json.loads(
        (DATA / "chunks.json").read_text(encoding="utf-8")
    )
    questions = json.loads(
        (DATA / "questions_seed.json").read_text(
            encoding="utf-8"
        )
    )

    dev = [
        question
        for question in questions
        if question["split"] == "dev"
    ]

    reviews = []
    resolved = 0

    for question in dev:
        evidence_checks = []

        for gold in question["gold"]:
            evidence = normalize(gold["evidence"])

            matches = [
                chunk
                for chunk in chunks
                if chunk["source_id"] == gold["source_id"]
                and chunk["page"] == gold["page"]
                and evidence
                and evidence in normalize(chunk["text"])
            ]

            evidence_checks.append({
                "gold": gold,
                "matching_chunk_ids": [
                    chunk["id"] for chunk in matches
                ],
                "matches": len(matches),
            })

        found = any(
            check["matches"] > 0
            for check in evidence_checks
        )
        resolved += int(found)

        reviews.append({
            "id": question["id"],
            "question": question["question"],
            "expected_answer": question["expected_answer"],
            "evidence_found_in_corpus": found,
            "evidence_checks": evidence_checks,
            "original_pdf_review": "pending",
            "alternative_evidence_review": "pending",
        })

        status = "FOUND" if found else "CHECK"
        print(f"{question['id']}: {status}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT / "dev_evidence_audit.json"

    output_path.write_text(
        json.dumps({
            "scope": "dev only",
            "normalization": "Unicode NFKC and whitespace",
            "resolved_questions": resolved,
            "total_questions": len(dev),
            "note": (
                "Matching text is a structural check, not proof "
                "that the answer is correct or the PDF extraction "
                "is faithful. Alternative evidence is not "
                "automatically discovered."
            ),
            "reviews": reviews,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nEvidence found: {resolved}/{len(dev)}")
    print("Original PDF review: still required.")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()