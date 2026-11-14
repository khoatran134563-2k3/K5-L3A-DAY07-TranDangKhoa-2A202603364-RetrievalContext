"""Prepare a 17-case dev benchmark draft for source review."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "starter/data/day6"


def main():
    questions = json.loads(
        (DATA / "questions_seed.json").read_text(
            encoding="utf-8"
        )
    )
    chunks = json.loads(
        (DATA / "chunks.json").read_text(encoding="utf-8")
    )
    chunk_by_id = {chunk["id"]: chunk for chunk in chunks}

    dev = [
        question
        for question in questions
        if question["split"] == "dev"
    ]

    if len(dev) != 15:
        raise ValueError("Expected 15 answerable dev questions.")

    q11 = next(question for question in dev if question["id"] == "q11")

    required = q11.get("required_chunk_ids", [])
    if len(required) != 2:
        raise ValueError("q11 must identify its two required chunks.")

    for chunk_id in required:
        if chunk_id not in chunk_by_id:
            raise ValueError(f"Missing q11 chunk: {chunk_id}")

    # Reconstruct the original quote from verified corpus wording.
    q11["gold"][0]["evidence"] = (
        "In developing countries, the ICT sector accounts for only "
        "about 1 percent of the workforce on average: less than "
        "0.5 percent in Bolivia and Ghana, and just under 2 percent "
        "in Colombia and Sri Lanka. In OECD countries, about "
        "3–5 percent of the employ- ment is in this sector."
    )

    # Add an alternative evidence location for q01.
    # This is evaluation data only, never an agent input.
    q01 = next(question for question in dev if question["id"] == "q01")
    alternative = {
        "source_id": "world_bank_wdr2016_overview",
        "page": 30,
        "evidence": (
            "The main ones are regulations that ensure a high "
            "degree of competition, skills that leverage technology, "
            "and institutions that are accountable (figure O.13)."
        ),
    }

    if alternative not in q01["gold"]:
        q01["gold"].append(alternative)

    unanswerable = [
        {
            "id": "dev_u01",
            "split": "dev",
            "question": (
                "How many public-sector ghost workers did Nigeria's "
                "e-ID identify during calendar year 2025?"
            ),
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": (
                "The indexed corpus is a 2016 report. Its historical "
                "Nigeria example does not establish the number "
                "identified during calendar year 2025."
            ),
            "absence_review_plan": (
                "Read all Nigeria, e-ID and ghost-worker passages "
                "in the indexed corpus; check dates and scope "
                "against the source PDF."
            ),
            "review_status": "needs_source_review",
        },
        {
            "id": "dev_u02",
            "split": "dev",
            "question": (
                "What was the average TFP growth advantage of "
                "Vietnamese firms using e-commerce during "
                "calendar year 2025?"
            ),
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": (
                "The indexed corpus is a 2016 report. Its historical "
                "Vietnamese e-commerce result does not establish "
                "an estimate for calendar year 2025."
            ),
            "absence_review_plan": (
                "Read all Vietnam, Vietnamese, e-commerce and TFP "
                "passages in the indexed corpus; check dates and "
                "scope against the source PDF."
            ),
            "review_status": "needs_source_review",
        },
    ]

    dev.extend(unanswerable)

    ids = [question["id"] for question in dev]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate dev question IDs.")

    output = DATA / "dev_draft.json"
    if output.exists():
        raise FileExistsError(
            f"Already exists; review before replacing: {output}"
        )

    output.write_text(
        json.dumps(dev, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Answerable dev cases: 15")
    print("Unanswerable dev cases: 2")
    print("Total dev cases: 17")
    print("q11: evidence spelling corrected in output.")
    print("q01: page 30 alternative added for source review.")
    print("Review status: pending original-source verification.")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()