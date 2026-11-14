"""Export original PDF pages for evidence review. No API calls."""

import json
from pathlib import Path

from pypdf import PdfReader


DAY7 = Path(__file__).resolve().parents[1]
DAY6 = Path(
    r"D:\VIN Lab\K5-L3A-DAY06-TranDangKhoa-2A202603364-DataIngestion"
)

RUNS = [
    "agent_dev_dense_baseline_20261114T100948307118Z",
    "agent_dev_tfidf_20261114T101205688015Z",
    "agent_dev_hybrid_rrf_20261114T102003609244Z",
    "agent_dev_dense_more_context_20261114T102334968926Z",
]


def main():
    pdf_path = (
        DAY6 / "starter/data"
        / "world_bank_digital_dividends_2016_overview.pdf"
    )

    questions = json.loads(
        (DAY7 / "starter/data/day6/dev_draft.json")
        .read_text(encoding="utf-8")
    )

    pages = set()

    # Include all declared dev gold locations.
    for question in questions:
        for gold in question.get("gold", []):
            pages.add(gold["page"])

    # Include pages actually supplied to the agents.
    for name in RUNS:
        traces = json.loads(
            (DAY7 / "starter/results" / name / "traces.json")
            .read_text(encoding="utf-8")
        )
        for trace in traces:
            for chunk in trace.get("context_chunks", []):
                pages.add(chunk["page"])

    reader = PdfReader(pdf_path)

    output = (
        DAY7 / "starter/results/source_review/source_pages.txt"
    )
    if output.exists():
        raise FileExistsError(f"Already exists: {output}")

    parts = [
        "SOURCE PDF TEXT — FOR REVIEW, NOT FINAL HUMAN LABELS",
        f"Source: {pdf_path.name}",
        f"Total PDF pages: {len(reader.pages)}",
        (
            "Extraction may lose layout or contain font/spacing errors. "
            "Compare ambiguous passages with the PDF visually."
        ),
    ]

    for page_number in sorted(pages):
        if not 1 <= page_number <= len(reader.pages):
            raise ValueError(f"Invalid PDF page: {page_number}")

        text = reader.pages[page_number - 1].extract_text() or ""
        parts.extend([
            "",
            f"===== PDF PAGE {page_number} =====",
            text,
        ])

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")

    print(f"Exported source pages: {len(pages)}")
    print(f"Saved: {output}")
    print("No API calls. Original PDF unchanged.")


if __name__ == "__main__":
    main()