"""Prepare review proposals from inspected dev traces. No API calls."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "starter/results"

RUNS = {
    "dense": "agent_dev_dense_baseline_20261114T100948307118Z",
    "tfidf": "agent_dev_tfidf_20261114T101205688015Z",
    "hybrid": "agent_dev_hybrid_rrf_20261114T102003609244Z",
    "dense_more": "agent_dev_dense_more_context_20261114T102334968926Z",
}

PAGES = {
    "q01": "14, 30",
    "q02": "15, 30, 31, 48",
    "q03": "16",
    "q04": "18",
    "q05": "19",
    "q06": "20",
    "q07": "21, 22",
    "q08": "23",
    "q09": "24",
    "q10": "25",
    "q11": "26",
    "q12": "27",
    "q13": "29",
    "q14": "31, 44",
    "q15": "34",
}

PARTIAL = {
    "dense": {"q02", "q11", "q15"},
    "tfidf": {"q11", "q15"},
    "hybrid": {"q11", "q15"},
    "dense_more": {"q15"},
}


def main():
    for name in RUNS.values():
        output = RESULTS / name / "reviews_proposed.json"
        if output.exists():
            raise FileExistsError(f"Already exists: {output}")

    for config, name in RUNS.items():
        directory = RESULTS / name
        traces = json.loads(
            (directory / "traces.json").read_text(encoding="utf-8")
        )

        reviews = []
        errors = []

        for trace in traces:
            qid = trace["query_id"]

            if trace["status"] == "error":
                errors.append({
                    "query_id": qid,
                    "error": trace.get("error"),
                })
                continue

            support = [True for _ in trace["citations"]]

            review = {
                "query_id": qid,
                "citation_support": support,
                "context_grade": None,
                "reason": "",
                "review_status": "assistant_proposal_pending_user_review",
                "review_basis": (
                    "Saved answer, actual context, citations and "
                    "exported original-PDF text. Not an LLM-judge score."
                ),
                "user_confirmed": False,
            }

            if qid.startswith("dev_u"):
                review["abstention_correct"] = (
                    trace["status"] == "abstain"
                    and not trace["citations"]
                )
                review["reason"] = (
                    "The final response declines to provide the requested "
                    "2025 statistic without substituting historical values. "
                    "Related source examples are in the 2016 report."
                )
                reviews.append(review)
                continue

            review["answer_grade"] = (
                1 if qid in PARTIAL[config] else 2
            )
            review["context_grade"] = 2
            review["reason"] = (
                f"Core answer agrees with the source text on PDF "
                f"page(s) {PAGES[qid]}; review the cited passage "
                "against the associated claims."
            )

            if qid == "q01" and config == "dense":
                review["citation_support"] = [
                    not cid.startswith("5442f7b3")
                    for cid in trace["citations"]
                ]
                review["reason"] = (
                    "Page 30 supports accountable institutions. "
                    "The additional cited page-16 chunk discusses analog "
                    "complements generally but does not state this condition."
                )

            if qid == "q02":
                review["reason"] = (
                    "Assess semantic evidence, not exact gold matching. "
                    "Page 30 supports concentration and monopolies; "
                    "page 15 explicitly mentions incumbent firms."
                )
                if config == "dense":
                    review["context_grade"] = 1
                    review["reason"] += (
                        " Dense cites dominance and antitrust passages, "
                        "but the absence-of-competition condition is less "
                        "directly established in its selected context."
                    )

            if qid == "q11":
                required = {
                    "3c5eced6c44e63d2c8bdaee49ea6bd411d8a3cb08bb9a94f617483a38d619fa9",
                    "d207fb8f11ee57fed9f67ade7fb88c930384c1dbbbc87451289b0ca28345440b",
                }
                complete = required.issubset(
                    set(trace["context_ids"])
                )
                review["context_grade"] = 2 if complete else 1

                # Under the declared evidence rule, one cited chunk
                # alone does not establish the preceding referent.
                review["citation_support"] = [
                    False for _ in trace["citations"]
                ]
                review["reason"] = (
                    "The 3–5% value is correct. The declared rule requires "
                    "both the ICT antecedent and employment-value chunks. "
                    f"Both in context: {complete}. "
                    "The final answer cites only the value chunk, so the "
                    "citation does not cover the complete evidence group."
                )

            if qid == "q15":
                review["citation_support"] = [
                    False for _ in trace["citations"]
                ]
                review["reason"] = (
                    "The main low-wage explanation is correct. "
                    "The answer additionally claims lower ICT use/adoption "
                    "or that manual nonroutine work is harder to automate. "
                    "The cited chunk does not fully support those additions. "
                    "False means incomplete support for associated claims, "
                    "not that the entire passage is irrelevant."
                )

            reviews.append(review)

        output = directory / "reviews_proposed.json"
        output.write_text(
            json.dumps({
                "status": "proposal_not_final_human_labels",
                "reviews": reviews,
                "errors_reported_separately": errors,
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        print(
            f"{config}: {len(reviews)} proposals, "
            f"{len(errors)} errors | {output}"
        )

    print("No API calls. Review proposals before finalizing.")


if __name__ == "__main__":
    main()