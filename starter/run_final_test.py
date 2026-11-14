"""Run frozen baseline and selected agent on the frozen test set."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from rag_agent import DAY6_STARTER, make_retriever, run_rag


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "benchmark/test_questions.json"
MANIFEST_PATH = ROOT / "benchmark/test_manifest.json"
SELECTION_PATH = ROOT / "artifacts/final_selection.json"
OUTPUT = ROOT / "starter/results/final_test"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_review_packet(directory, questions, traces):
    by_id = {q["id"]: q for q in questions}
    lines = ["# Test review — pending manual grading", ""]

    for trace in traces:
        q = by_id[trace["query_id"]]
        lines.extend([
            f"## {q['id']}",
            "",
            f"Question: {q['question']}",
            f"Answerable: {q['answerable']}",
            f"Status: {trace['status']}",
            "",
            "### Agent answer",
            trace.get("answer") or "(No final answer)",
            "",
            "Citations: " + json.dumps(trace.get("citations", [])),
            "",
        ])

        if trace["status"] == "error":
            lines.extend(["Error: " + trace.get("error", ""), ""])

        lines.extend(["### Evaluation reference — not agent input", ""])

        if q["answerable"]:
            lines.extend([
                q["expected_answer"],
                "```json",
                json.dumps(q["gold"], ensure_ascii=False, indent=2),
                "```",
                "",
            ])
        else:
            lines.extend([q["absence_reason"], ""])

        lines.extend(["### Actual final context", ""])
        for chunk in trace.get("context_chunks", []):
            lines.extend([
                f"Chunk: {chunk['id']}",
                f"Source: {chunk['source']} | PDF page: {chunk['page']}",
                chunk["text"],
                "",
            ])

        lines.extend(["---", ""])

    (directory / "review_packet.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    manifest = load(MANIFEST_PATH)
    selection = load(SELECTION_PATH)
    questions = load(TEST_PATH)

    if sha256(TEST_PATH) != manifest["test_sha256"]:
        raise ValueError("Frozen test changed.")
    if sha256(SELECTION_PATH) != manifest["selection_sha256"]:
        raise ValueError("Frozen selection changed.")
    if sha256(ROOT / "starter/data/day6/chunks.json") != manifest["corpus_sha256"]:
        raise ValueError("Corpus changed.")

    for name, expected in manifest["code_sha256"].items():
        if sha256(ROOT / "starter" / name) != expected:
            raise ValueError(f"Frozen code changed: {name}")

    if (
        len(questions) != 7
        or any(q["split"] != "test" for q in questions)
        or sum(q["answerable"] for q in questions) != 5
        or len({q["id"] for q in questions}) != 7
    ):
        raise ValueError("Expected 7 unique test cases: 5 + 2.")

    if OUTPUT.exists():
        raise FileExistsError(
            "final_test already exists. Preserve the first test run; "
            "do not overwrite or rerun to select better outputs."
        )

    load_dotenv(DAY6_STARTER / ".env", override=False)
    OUTPUT.mkdir(parents=True, exist_ok=False)

    save(OUTPUT / "execution_manifest.json", {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_manifest": manifest,
        "runner_sha256": sha256(Path(__file__)),
        "note": (
            "Original frozen test run. Baseline runs first; query "
            "embedding cache may benefit the second configuration. "
            "Report errors and do not tune using these outputs."
        ),
    })

    all_summaries = []

    for role in ("baseline", "selected"):
        frozen = selection[role]
        name = frozen["name"]
        config = frozen["config"]
        directory = OUTPUT / name
        directory.mkdir()

        save(directory / "protocol.json", {
            "split": "test",
            "role": role,
            "model": selection["model"],
            "config": frozen,
            "test_sha256": manifest["test_sha256"],
            "code_sha256": manifest["code_sha256"],
        })

        save(directory / "cases.json", [
            {
                "id": q["id"],
                "split": "test",
                "answerable": q["answerable"],
            }
            for q in questions
        ])

        retriever = make_retriever(config["method"], config["top_k"])
        traces = []
        reviews = []
        errors = []

        for index, q in enumerate(questions, start=1):
            # Pass ONLY the question, never gold or answerability.
            trace = run_rag(
                q["question"],
                retriever,
                model=selection["model"],
                top_k=config["top_k"],
                budget=config["budget"],
                max_searches=config["max_searches"],
            )
            trace["query_id"] = q["id"]
            trace["question"] = q["question"]
            traces.append(trace)

            if trace["status"] == "error":
                errors.append({
                    "query_id": q["id"],
                    "error": trace.get("error"),
                })
            else:
                review = {
                    "query_id": q["id"],
                    "citation_support": [
                        None for _ in trace["citations"]
                    ],
                    "context_grade": None,
                    "reason": "",
                    "review_status": "pending",
                    "user_confirmed": False,
                }

                if q["answerable"]:
                    review["answer_grade"] = None
                else:
                    review["abstention_correct"] = None

                reviews.append(review)

            save(directory / "traces.json", traces)
            save(directory / "reviews_proposed.json", {
                "status": "pending_manual_test_review",
                "reviews": reviews,
                "errors_reported_separately": errors,
            })

            print(
                f"{name} | {index}/7 | {q['id']} "
                f"| {trace['status']} "
                f"| searches={trace['search_count']}",
                flush=True,
            )

        write_review_packet(directory, questions, traces)

        model_events = [
            event
            for trace in traces
            for event in trace["events"]
            if event["event"] == "model"
        ]

        summary = {
            "config": name,
            "cases": len(traces),
            "answered_status": sum(t["status"] == "answered" for t in traces),
            "abstain_status": sum(t["status"] == "abstain" for t in traces),
            "execution_errors": len(errors),
            "model_prompt_tokens": sum(
                e.get("usage", {}).get("prompt_tokens", 0)
                for e in model_events
            ),
            "model_completion_tokens": sum(
                e.get("usage", {}).get("completion_tokens", 0)
                for e in model_events
            ),
            "elapsed_seconds": sum(t["elapsed_seconds"] for t in traces),
            "quality_metrics": "Pending manual test review.",
        }

        save(directory / "run_summary.json", summary)
        all_summaries.append(summary)

    save(OUTPUT / "run_summaries.json", all_summaries)
    print("\nTEST RUN SUMMARIES — NOT QUALITY SCORES")
    print(json.dumps(all_summaries, ensure_ascii=False, indent=2))
    print(f"\nSaved: {OUTPUT}")
    print("Do not tune using test results. Grade saved outputs next.")


if __name__ == "__main__":
    main()