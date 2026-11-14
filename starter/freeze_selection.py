"""Freeze the dev-based selection before test. No API calls."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "starter/results"

RUNS = {
    "dense_baseline": "agent_dev_dense_baseline_20261114T100948307118Z",
    "tfidf": "agent_dev_tfidf_20261114T101205688015Z",
    "hybrid_rrf": "agent_dev_hybrid_rrf_20261114T102003609244Z",
    "dense_more_context": "agent_dev_dense_more_context_20261114T102334968926Z",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = ROOT / "artifacts/final_selection.json"

    if output.exists():
        raise FileExistsError(
            f"Selection already exists; do not overwrite: {output}"
        )

    protocols = {}
    summaries = {}
    review_hashes = {}

    for config, run in RUNS.items():
        directory = RESULTS / run

        protocols[config] = read_json(directory / "protocol.json")
        summaries[config] = read_json(
            directory / "evaluated_summary.json"
        )
        reviews = read_json(directory / "reviews.json")

        if any(
            review.get("review_status") != "user_reviewed"
            or review.get("user_confirmed") is not True
            for review in reviews
        ):
            raise ValueError(f"{config}: unconfirmed reviews.")

        review_hashes[config] = digest(directory / "reviews.json")

    selected_protocol = protocols["hybrid_rrf"]
    expected_code = selected_protocol["code_sha256"]
    expected_benchmark = selected_protocol["benchmark_sha256"]

    for config, protocol in protocols.items():
        if (
            protocol["code_sha256"] != expected_code
            or protocol["benchmark_sha256"] != expected_benchmark
        ):
            raise ValueError(
                f"{config}: code or benchmark differs between runs."
            )

    # Ensure current implementation still matches the compared runs.
    for filename, expected in expected_code.items():
        if digest(ROOT / "starter" / filename) != expected:
            raise ValueError(
                f"Code changed after dev: {filename}. "
                "Review the change before freezing."
            )

    benchmark = ROOT / "starter/data/day6/dev_draft.json"
    if digest(benchmark) != expected_benchmark:
        raise ValueError("Dev benchmark changed after the compared runs.")

    record = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_basis": "Reviewed dev results only; test not run.",
        "baseline": {
            "name": "dense_baseline",
            "config": protocols["dense_baseline"]["config"],
        },
        "selected": {
            "name": "hybrid_rrf",
            "config": selected_protocol["config"],
            "candidate_k_per_method": 10,
            "rrf_k": 60,
        },
        "model": selected_protocol["model"],
        "reason": [
            "Hybrid has one execution error, versus three for TF-IDF "
            "and two for dense_more_context.",
            "Both unanswerable dev cases were correctly declined.",
            "Dense_more_context uses substantially more input tokens; "
            "its higher conditional answer mean excludes two errors.",
            "Conditional quality means have different denominators "
            "and are not overall accuracy.",
        ],
        "known_limitations": [
            "Hybrid q14 failed quotation validation.",
            "Hybrid q11 lacked its declared complete evidence group.",
            "Hybrid q15 included an insufficiently supported addition.",
            "LLM verification does not replace source-based review.",
            "Small dev set; selection does not establish general superiority.",
        ],
        "dev_runs": RUNS,
        "dev_summaries": summaries,
        "review_sha256": review_hashes,
        "code_sha256": expected_code,
        "dev_benchmark_sha256": expected_benchmark,
        "corpus_sha256": digest(
            ROOT / "starter/data/day6/chunks.json"
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Baseline: dense_baseline")
    print("Selected: hybrid_rrf")
    print("Dev reviews and implementation hashes checked.")
    print(f"Saved: {output}")
    print("No API calls. Test has not been run.")


if __name__ == "__main__":
    main()