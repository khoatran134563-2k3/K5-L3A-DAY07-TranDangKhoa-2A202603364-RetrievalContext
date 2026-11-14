"""Validate confirmed reviews and aggregate saved results. No API calls."""

import argparse
import json
from collections import Counter
from pathlib import Path

from agent_metrics import aggregate


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()

    directory = args.run
    cases = load_json(directory / "cases.json")
    traces = load_json(directory / "traces.json")
    proposal = load_json(directory / "reviews_proposed.json")
    reviews = proposal["reviews"]

    case_by_id = {case["id"]: case for case in cases}
    trace_by_id = {trace["query_id"]: trace for trace in traces}
    review_ids = [review["query_id"] for review in reviews]

    if len(case_by_id) != len(cases):
        raise ValueError("Duplicate case IDs.")
    if len(trace_by_id) != len(traces):
        raise ValueError("Duplicate trace IDs.")
    if len(set(review_ids)) != len(review_ids):
        raise ValueError("Duplicate review IDs.")
    if set(case_by_id) != set(trace_by_id):
        raise ValueError("Case and trace IDs do not match.")

    expected_ids = {
        cid for cid, trace in trace_by_id.items()
        if trace["status"] != "error"
    }
    if set(review_ids) != expected_ids:
        raise ValueError(
            "Reviews must cover exactly the non-error cases."
        )

    for review in reviews:
        cid = review["query_id"]
        case = case_by_id[cid]
        trace = trace_by_id[cid]

        if (
            review.get("review_status") != "user_reviewed"
            or review.get("user_confirmed") is not True
        ):
            raise ValueError(f"{cid}: review not confirmed.")

        reason = review.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{cid}: missing review reason.")

        support = review.get("citation_support")
        if (
            not isinstance(support, list)
            or len(support) != len(trace["citations"])
            or any(type(value) is not bool for value in support)
        ):
            raise ValueError(f"{cid}: invalid citation labels.")

        if case["answerable"]:
            context_grade = review.get("context_grade")
            if (
                type(context_grade) is not int
                or context_grade not in (0, 1, 2)
            ):
                raise ValueError(f"{cid}: invalid context grade.")

    # Starter validates answer grades, abstention and trace structure.
    summary = aggregate(cases, traces, reviews)

    # Starter calls every error "provider_errors", although our
    # traces also include validation errors. Use an accurate name.
    summary["execution_errors"] = summary.pop("provider_errors")

    errors = [
        {
            "query_id": trace["query_id"],
            "error_type": trace.get("error_type"),
            "error": trace.get("error"),
        }
        for trace in traces
        if trace["status"] == "error"
    ]

    context_counts = Counter(
        review["context_grade"]
        for review in reviews
        if case_by_id[review["query_id"]]["answerable"]
    )

    summary["context_review_non_error_cases"] = {
        "0": context_counts[0],
        "1": context_counts[1],
        "2": context_counts[2],
        "N": sum(context_counts.values()),
        "note": (
            "Only reviewed non-error answerable cases. "
            "This is not the context rate for the entire benchmark."
        ),
    }
    summary["errors_detail"] = errors
    summary["quality_denominator_note"] = (
        "Starter quality metrics exclude execution-error cases. "
        "Report execution errors alongside quality metrics; "
        "do not describe the conditional mean as overall accuracy."
    )

    reviews_path = directory / "reviews.json"
    summary_path = directory / "evaluated_summary.json"

    for path in (reviews_path, summary_path):
        if path.exists():
            raise FileExistsError(
                f"Already exists; review before replacing: {path}"
            )

    save_json(reviews_path, reviews)
    save_json(summary_path, summary)

    print(f"Confirmed reviews: {len(reviews)}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved labels: {reviews_path}")
    print(f"Saved summary: {summary_path}")
    print("No API calls.")


if __name__ == "__main__":
    main()