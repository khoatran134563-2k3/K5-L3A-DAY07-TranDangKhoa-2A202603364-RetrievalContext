"""Prepare and freeze a new seven-case test set. No API calls."""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "starter/data/day6/chunks.json"
SELECTION_PATH = ROOT / "artifacts/final_selection.json"
OUTPUT_DIR = ROOT / "benchmark"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def main():
    selection = json.loads(
        SELECTION_PATH.read_text(encoding="utf-8")
    )
    chunks = json.loads(
        CORPUS_PATH.read_text(encoding="utf-8")
    )

    if sha256(CORPUS_PATH) != selection["corpus_sha256"]:
        raise ValueError("Corpus changed after selection.")

    for name, expected in selection["code_sha256"].items():
        if sha256(ROOT / "starter" / name) != expected:
            raise ValueError(f"Implementation changed: {name}")

    def answerable(cid, question, answer, page, evidence):
        matches = [
            chunk for chunk in chunks
            if chunk["page"] == page
            and normalize(evidence) in normalize(chunk["text"])
        ]

        if not matches:
            raise ValueError(f"{cid}: evidence not found in corpus.")

        source_ids = {chunk["source_id"] for chunk in matches}
        if len(source_ids) != 1:
            raise ValueError(f"{cid}: ambiguous source.")

        return {
            "id": cid,
            "split": "test",
            "question": question,
            "answerable": True,
            "expected_answer": answer,
            "gold": [{
                "source_id": matches[0]["source_id"],
                "page": page,
                "evidence": evidence,
            }],
            "filter": {},
            "evidence_mode": "single_chunk_alternatives",
            "matching_chunk_ids": [
                chunk["id"] for chunk in matches
            ],
            "source_review_basis": (
                "Corpus evidence and exported original-PDF text."
            ),
        }

    questions = [
        answerable(
            "t01",
            "According to the report, by 2014 how many UN member "
            "states' national websites enabled citizens to create "
            "personal online accounts?",
            "101.",
            18,
            "By 2014, all 193 member states of the United Nations "
            "(UN) had national websites: 101 enabled citizens to "
            "create per- sonal online accounts, 73 to file income "
            "taxes, and 60 to register a business.",
        ),
        answerable(
            "t02",
            "By approximately how much did sophisticated reservation "
            "and pricing algorithms increase load factors for U.S. "
            "domestic flights between 1993 and 2007?",
            "By about one-third.",
            24,
            "In the airline industry, sophisticated reservation and "
            "pricing algorithms increased load factors by about "
            "one-third for U.S. domestic flights between 1993 and 2007.",
        ),
        answerable(
            "t03",
            "According to the report, by up to what percentage did "
            "TransferWise and Xoom cut the prices of international "
            "currency transfers?",
            "Up to 90 percent.",
            26,
            "Similarly, TransferWise and Xoom have reduced regulatory "
            "rents in the financial sector and cut the prices of "
            "international currency transfers by up to 90 percent.",
        ),
        answerable(
            "t04",
            "How many employees did Instagram have in 2012 when "
            "Facebook bought it?",
            "13 employees.",
            26,
            "Instagram, a photo sharing app, had just 13 employees "
            "in 2012 when it was bought by Facebook for US$1 billion.",
        ),
        answerable(
            "t05",
            "In Pakistan, by what percentage range did the shift "
            "enabled by mobile phones reduce postharvest losses "
            "from the most perishable crops?",
            "By 21–35 percent.",
            27,
            "In Pakistan, mobile phones allow farmers to shift to "
            "more perishable but higher return cash crops, reducing "
            "postharvest losses from the most perishable crops "
            "by 21–35 percent.",
        ),
        {
            "id": "test_u01",
            "split": "test",
            "question": (
                "What exact implementation source code does the "
                "indexed report provide for UPS's routing algorithm "
                "that avoids left turns?"
            ),
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": (
                "The indexed corpus describes the UPS routing practice "
                "and its benefits, but does not supply implementation "
                "source code. General routing knowledge or invented "
                "code would not answer a request for code provided "
                "by this source."
            ),
            "absence_check": {
                "scope": "Indexed 132-chunk Day 6 strategy B corpus.",
                "terms_reviewed": [
                    "UPS", "routing", "source code", "pseudocode"
                ],
                "relevant_pdf_pages": [24],
            },
        },
        {
            "id": "test_u02",
            "split": "test",
            "question": (
                "What are the full names of all 62,000 public-sector "
                "ghost workers identified by Nigeria's e-ID in "
                "the indexed report?"
            ),
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": (
                "The indexed corpus gives the aggregate count and "
                "annual saving, but contains no full list of individual "
                "names. Repeating the count is not an answer to the "
                "requested list."
            ),
            "absence_check": {
                "scope": "Indexed 132-chunk Day 6 strategy B corpus.",
                "terms_reviewed": [
                    "Nigeria", "e-ID", "ghost workers"
                ],
                "relevant_pdf_pages": [29],
            },
        },
    ]

    seed = json.loads(
        (ROOT / "starter/data/day6/questions_seed.json")
        .read_text(encoding="utf-8")
    )
    dev = json.loads(
        (ROOT / "starter/data/day6/dev_draft.json")
        .read_text(encoding="utf-8")
    )

    old_questions = {
        normalize(q["question"]).casefold()
        for q in seed + dev
    }

    for question in questions:
        if normalize(question["question"]).casefold() in old_questions:
            raise ValueError(f"Duplicate question: {question['id']}")

    if (
        len(questions) != 7
        or sum(q["answerable"] for q in questions) != 5
        or len({q["id"] for q in questions}) != 7
    ):
        raise ValueError("Expected five answerable and two unanswerable cases.")

    test_path = OUTPUT_DIR / "test_questions.json"
    manifest_path = OUTPUT_DIR / "test_manifest.json"

    for path in (test_path, manifest_path):
        if path.exists():
            raise FileExistsError(
                f"Already exists; do not overwrite frozen test: {path}"
            )

    # Construct exact bytes before writing and hash those bytes.
    test_text = (
        json.dumps(questions, ensure_ascii=False, indent=2) + "\n"
    )
    test_bytes = test_text.encode("utf-8")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "test",
        "answerable": 5,
        "unanswerable": 2,
        "test_sha256": hashlib.sha256(test_bytes).hexdigest(),
        "selection_sha256": sha256(SELECTION_PATH),
        "corpus_sha256": sha256(CORPUS_PATH),
        "model": selection["model"],
        "baseline": selection["baseline"],
        "selected": selection["selected"],
        "code_sha256": selection["code_sha256"],
        "test_execution_status_at_creation": "not_run",
        "grading": {
            "answer_grade": "0 incorrect/unsupported; 1 partial; 2 correct and complete",
            "citation_support": (
                "Check each citation against the actual context "
                "and associated claims."
            ),
            "abstention": (
                "Decline the unsupported request without inventing "
                "code or names or substituting related aggregate facts."
            ),
            "context_grade": (
                "For answerable cases: 0 absent; 1 incomplete; "
                "2 sufficient evidence and source."
            ),
        },
        "limitations": [
            "Small test set, mostly single-passage factual questions.",
            "Some pages/chunks overlap dev, but requested facts differ.",
            "Exact-question duplication is checked by code; "
            "semantic distinction was reviewed during preparation.",
            "No gold or labels may enter the agent or retrieval.",
            "Do not tune the implementation using test outputs.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_path.write_bytes(test_bytes)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Answerable test cases: 5")
    print("Unanswerable test cases: 2")
    print("Gold evidence found in the shared corpus.")
    print("Implementation matches the frozen selection.")
    print(f"Saved: {test_path}")
    print(f"Saved: {manifest_path}")
    print("No API calls. Test has not been run.")


if __name__ == "__main__":
    main()