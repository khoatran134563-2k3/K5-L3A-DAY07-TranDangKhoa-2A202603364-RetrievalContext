"""Create readable review packets from saved runs. No API calls."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "starter/results"

RUNS = [
    "agent_dev_dense_baseline_20261114T100948307118Z",
    "agent_dev_tfidf_20261114T101205688015Z",
    "agent_dev_hybrid_rrf_20261114T102003609244Z",
    "agent_dev_dense_more_context_20261114T102334968926Z",
]


def main():
    questions = json.loads(
        (ROOT / "starter/data/day6/dev_draft.json")
        .read_text(encoding="utf-8")
    )
    by_id = {question["id"]: question for question in questions}

    # Refuse to overwrite packets containing manual review work.
    for name in RUNS:
        output = RESULTS / name / "review_packet.md"
        if output.exists():
            raise FileExistsError(f"Already exists: {output}")

    for name in RUNS:
        directory = RESULTS / name
        traces = json.loads(
            (directory / "traces.json").read_text(encoding="utf-8")
        )

        lines = [
            f"# Review: {name}",
            "",
            "Trạng thái: chờ chấm và đối chiếu PDF gốc.",
            "Gold bên dưới chỉ dùng để đánh giá, không gửi vào agent.",
            "",
        ]

        for trace in traces:
            question = by_id[trace["query_id"]]

            lines.extend([
                f"## {question['id']}",
                "",
                f"Question: {question['question']}",
                f"Answerable: {question['answerable']}",
                f"Agent status: {trace['status']}",
                "",
                "### Câu trả lời",
                "",
                trace.get("answer") or "(Không có câu trả lời cuối)",
                "",
                "Citations:",
                json.dumps(trace.get("citations", [])),
                "",
            ])

            if trace["status"] == "error":
                lines.extend([
                    f"Error: {trace.get('error', '')}",
                    "",
                ])

            if question["answerable"]:
                lines.extend([
                    "### Đáp án và evidence chuẩn đang rà soát",
                    "",
                    question["expected_answer"],
                    "",
                    "```json",
                    json.dumps(
                        question["gold"],
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "```",
                    "",
                ])

                if question.get("required_chunk_ids"):
                    lines.extend([
                        "Required chunks:",
                        json.dumps(question["required_chunk_ids"]),
                        "",
                    ])
            else:
                lines.extend([
                    "### Lý do thiếu bằng chứng cần kiểm tra",
                    "",
                    question["absence_reason"],
                    question.get("absence_review_plan", ""),
                    "",
                ])

            lines.extend([
                "### Context thực tế",
                "",
            ])

            for chunk in trace.get("context_chunks", []):
                lines.extend([
                    f"#### Chunk {chunk['id']}",
                    f"Source: {chunk['source']}",
                    f"PDF page: {chunk['page']}",
                    "",
                    chunk["text"],
                    "",
                ])

            lines.extend([
                "### Nhãn chấm thủ công",
                "",
                "- Context grade 0/1/2: pending",
                "- Answer grade 0/1/2 nếu có đáp án: pending",
                "- Citation support theo từng citation: pending",
                "- Abstention correct nếu thiếu evidence: pending",
                "- Đã đối chiếu trang PDF gốc: pending",
                "- Lý do: pending",
                "",
                "---",
                "",
            ])

        output = directory / "review_packet.md"
        output.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        print(f"Prepared {len(traces)} cases: {output}")

    print("Review packets ready. No API calls.")


if __name__ == "__main__":
    main()