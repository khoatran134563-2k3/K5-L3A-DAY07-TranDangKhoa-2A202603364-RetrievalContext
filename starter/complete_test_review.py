"""Finalize the two test reviews after explicit user review. No API."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "starter/results/final_test"

REASONS = {
    "t01": "Trang PDF 18 hỗ trợ 101 website cho phép tạo tài khoản cá nhân.",
    "t02": "Trang PDF 24 hỗ trợ mức tăng khoảng một phần ba, đúng phạm vi chuyến bay và thời gian.",
    "t03": "Trang PDF 26 hỗ trợ TransferWise và Xoom giảm phí tối đa 90%.",
    "t04": "Trang PDF 26 hỗ trợ Instagram có 13 nhân viên khi Facebook mua năm 2012.",
    "t05": "Trang PDF 27 hỗ trợ mức giảm tổn thất sau thu hoạch 21–35% tại Pakistan.",
    "test_u01": "Câu trả lời từ chối cung cấp mã nguồn không có trong nguồn, không bịa code.",
    "test_u02": "Câu trả lời từ chối cung cấp danh sách tên không có trong nguồn, không bịa tên.",
}

EXPECTED_CITATIONS = {
    "t01": "b2ce1cceb9601d97eba8a2cc2a27f5ecb88bb26a618a0dcb989ace7ac0465a1c",
    "t02": "210ca4fd9cbc77a1d396c5803e4980f5e6bd01cb6b8ceb4401ed14e5163e948d",
    "t03": "ee71dee9c6d9234c0d32c8555570da2c4cfe54e6f1dce5e707716b02045bdd49",
    "t04": "d207fb8f11ee57fed9f67ade7fb88c930384c1dbbbc87451289b0ca28345440b",
    "t05": "654ee5e650f0fa16332bc041ce8cdf4a3765da63c7b4d354873601021b6a0507",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirm-reviewed",
        action="store_true",
        help="Confirm that you reviewed both test packets.",
    )
    args = parser.parse_args()

    if not args.confirm_reviewed:
        raise SystemExit(
            "Review both test packets before using --confirm-reviewed."
        )

    prepared = []

    # Validate both runs before writing labels.
    for name in ("dense_baseline", "hybrid_rrf"):
        directory = BASE / name
        traces = json.loads(
            (directory / "traces.json").read_text(encoding="utf-8")
        )

        if len(traces) != 7 or {t["query_id"] for t in traces} != set(REASONS):
            raise ValueError(f"{name}: unexpected test cases.")

        for filename in ("reviews.json", "evaluated_summary.json"):
            if (directory / filename).exists():
                raise FileExistsError(
                    f"{name}: {filename} already exists; do not overwrite."
                )

        reviews = []

        for trace in traces:
            cid = trace["query_id"]
            review = {
                "query_id": cid,
                "reason": REASONS[cid],
                "review_status": "user_reviewed",
                "user_confirmed": True,
                "review_basis": (
                    "User-confirmed comparison of saved test answer, "
                    "actual context and source evidence."
                ),
            }

            if cid in EXPECTED_CITATIONS:
                expected = EXPECTED_CITATIONS[cid]
                if (
                    trace["status"] != "answered"
                    or trace["citations"] != [expected]
                    or expected not in trace["context_ids"]
                    or f"[{expected}]" not in trace["answer"]
                ):
                    raise ValueError(f"{name}/{cid}: differs from reviewed output.")

                review.update({
                    "answer_grade": 2,
                    "context_grade": 2,
                    "citation_support": [True],
                })
            else:
                if trace["status"] != "abstain" or trace["citations"]:
                    raise ValueError(f"{name}/{cid}: unexpected abstention.")

                review.update({
                    "context_grade": None,
                    "citation_support": [],
                    "abstention_correct": True,
                })

            reviews.append(review)

        prepared.append((directory, reviews))

    for directory, reviews in prepared:
        path = directory / "reviews_proposed.json"
        path.write_text(
            json.dumps({
                "status": "user_reviewed",
                "reviews": reviews,
                "errors_reported_separately": [],
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        subprocess.run([
            sys.executable, "-X", "utf8",
            str(ROOT / "starter/finalize_reviews.py"),
            "--run", str(directory),
        ], check=True)

    print("\nBoth test reviews finalized. No API calls.")


if __name__ == "__main__":
    main()