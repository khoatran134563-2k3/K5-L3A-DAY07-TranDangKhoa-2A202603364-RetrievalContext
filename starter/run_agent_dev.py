"""Run the real RAG agent on dev and prepare human review labels."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from rag_agent import (
    MODEL,
    DAY6_STARTER,
    make_retriever,
    run_rag,
)


ROOT = Path(__file__).resolve().parents[1]

CONFIGS = {
    "dense_baseline": {
        "method": "dense",
        "top_k": 3,
        "budget": 2600,
        "max_searches": 3,
    },
    "tfidf": {
        "method": "tfidf",
        "top_k": 3,
        "budget": 2600,
        "max_searches": 3,
    },
    "hybrid_rrf": {
        "method": "hybrid",
        "top_k": 3,
        "budget": 2600,
        "max_searches": 3,
    },
    "dense_more_context": {
        "method": "dense",
        "top_k": 6,
        "budget": 5200,
        "max_searches": 3,
    },
}


def save_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        choices=list(CONFIGS),
        default="dense_baseline",
    )
    args = parser.parse_args()

    load_dotenv(DAY6_STARTER / ".env", override=False)

    benchmark_path = ROOT / "starter/data/day6/dev_draft.json"
    questions = json.loads(
        benchmark_path.read_text(encoding="utf-8")
    )

    if (
        len(questions) != 17
        or any(q["split"] != "dev" for q in questions)
        or sum(q["answerable"] for q in questions) != 15
        or len({q["id"] for q in questions}) != len(questions)
    ):
        raise ValueError("Expected 17 unique dev cases: 15 + 2.")

    config = CONFIGS[args.config]
    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    output_dir = (
        ROOT / "starter/results"
        / f"agent_dev_{args.config}_{stamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    # Record the exact local code version before running.
    code_hashes = {}
    for name in (
        "rag_agent.py",
        "dense_retrieval.py",
        "lexical_retrieval.py",
        "hybrid_retrieval.py",
        "benchmark_retrieval.py",
        "run_agent_dev.py",
    ):
        path = ROOT / "starter" / name
        code_hashes[name] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

    save_json(output_dir / "protocol.json", {
        "split": "dev",
        "config_name": args.config,
        "config": config,
        "model": MODEL,
        "benchmark_sha256": hashlib.sha256(
            benchmark_path.read_bytes()
        ).hexdigest(),
        "code_sha256": code_hashes,
        "source_review_status": "pending",
        "purpose": "Real-agent dev run before further improvements.",
        "notes": [
            "Only question text enters run_rag.",
            "Gold and answerability labels are not passed to the agent.",
            "Human review is required; statuses are not quality scores.",
            "Provider errors are retained separately.",
        ],
    })

    save_json(output_dir / "cases.json", [
        {
            "id": q["id"],
            "split": "dev",
            "answerable": q["answerable"],
        }
        for q in questions
    ])

    retriever = make_retriever(
        config["method"], config["top_k"]
    )

    traces = []
    reviews = []

    for index, question in enumerate(questions, start=1):
        # No expected_answer, gold or required IDs enter the agent.
        trace = run_rag(
            question["question"],
            retriever,
            model=MODEL,
            top_k=config["top_k"],
            budget=config["budget"],
            max_searches=config["max_searches"],
        )
        trace["query_id"] = question["id"]
        trace["question"] = question["question"]
        traces.append(trace)

        if trace["status"] != "error":
            review = {
                "query_id": question["id"],
                "citation_support": [
                    None for _ in trace["citations"]
                ],
                "context_grade": None,
                "reason": "",
                "review_status": "pending",
            }

            if question["answerable"]:
                review["answer_grade"] = None
            else:
                review["abstention_correct"] = None

            reviews.append(review)

        # Preserve progress after every completed case.
        save_json(output_dir / "traces.json", traces)
        save_json(output_dir / "reviews_pending.json", reviews)

        print(
            f"{index:02d}/{len(questions)}"
            f" | {question['id']}"
            f" | {trace['status']}"
            f" | searches={trace['search_count']}"
        )

    successful = [
        trace for trace in traces
        if trace["status"] != "error"
    ]
    model_events = [
        event
        for trace in traces
        for event in trace["events"]
        if event["event"] == "model"
    ]

    summary = {
        "config": args.config,
        "cases": len(traces),
        "answered_status": sum(
            t["status"] == "answered" for t in traces
        ),
        "abstain_status": sum(
            t["status"] == "abstain" for t in traces
        ),
        "errors": sum(
            t["status"] == "error" for t in traces
        ),
        "tool_use": {
            "n": sum(t["search_count"] > 0 for t in successful),
            "N": len(successful),
        },
        "model_responses_received": len(model_events),
        "model_prompt_tokens": sum(
            e.get("usage", {}).get("prompt_tokens", 0)
            for e in model_events
        ),
        "model_completion_tokens": sum(
            e.get("usage", {}).get("completion_tokens", 0)
            for e in model_events
        ),
        "citation_validation_failures": sum(
            event["event"] == "citation_validation_failed"
            for trace in traces
            for event in trace["events"]
        ),
        "elapsed_seconds": sum(
            t["elapsed_seconds"] for t in traces
        ),
        "quality_metrics": "Pending human review.",
    }

    save_json(output_dir / "run_summary.json", summary)

    print("\nRUN SUMMARY — NOT QUALITY SCORES")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()