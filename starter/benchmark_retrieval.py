"""Preliminary dev retrieval benchmark. Does not run an answer model."""

import hashlib
import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from dense_retrieval import DenseRetriever
from lexical_retrieval import LexicalRetriever
from hybrid_retrieval import HybridRetriever


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "starter/data/day6"

CONFIGS = [
    {
        "name": "dense_baseline",
        "method": "dense",
        "top_k": 3,
        "budget": 2600,
        "hypothesis": "Baseline.",
    },
    {
        "name": "tfidf",
        "method": "tfidf",
        "top_k": 3,
        "budget": 2600,
        "hypothesis": "Lexical matching may retrieve precise terms.",
    },
    {
        "name": "hybrid_rrf",
        "method": "hybrid",
        "top_k": 3,
        "budget": 2600,
        "hypothesis": "Combining rankings may improve evidence retrieval.",
    },
    {
        "name": "dense_more_context",
        "method": "dense",
        "top_k": 6,
        "budget": 5200,
        "hypothesis": "More chunks may retain supporting evidence.",
    },
]


def normalize(text):
    return re.sub(
        r"\s+", " ", unicodedata.normalize("NFKC", text)
    ).strip()


def has_evidence(question, chunks):
    if question.get("evidence_mode") == "all_required_chunks":
        required = set(question["required_chunk_ids"])
        return bool(required) and required.issubset(
            {chunk["id"] for chunk in chunks}
        )

    for gold in question["gold"]:
        evidence = normalize(gold["evidence"])
        if not evidence:
            continue

        for chunk in chunks:
            if (
                chunk["source_id"] == gold["source_id"]
                and chunk["page"] == gold["page"]
                and evidence in normalize(chunk["text"])
            ):
                return True

    return False


def pack_context(hits, budget):
    blocks = []
    selected = []
    seen = set()
    used = 0

    for hit in hits:
        if hit["id"] in seen:
            continue
        seen.add(hit["id"])

        block = (
            f"[{hit['id']}]\n"
            f"Source: {hit['source']}\n"
            f"PDF page: {hit['page']}\n"
            f"{hit['text']}"
        )

        extra = len(block) + (2 if blocks else 0)

        if used + extra <= budget:
            blocks.append(block)
            selected.append(hit)
            used += extra

    return "\n\n".join(blocks), selected


def rate(n, total):
    return {
        "n": n,
        "N": total,
        "rate": n / total if total else None,
    }


def group_metrics(records):
    result = {"questions": len(records)}

    for name in ("hit_at_1", "hit_at_3", "context_evidence"):
        result[name] = rate(
            sum(record.get(name, False) for record in records),
            len(records),
        )

    return result


def main():
    benchmark_path = DATA / "dev_draft.json"
    raw = benchmark_path.read_bytes()
    questions = json.loads(raw.decode("utf-8"))

    if (
        len(questions) != 17
        or any(question["split"] != "dev" for question in questions)
        or sum(q["answerable"] for q in questions) != 15
    ):
        raise ValueError("Expected dev-only benchmark: 15 + 2 cases.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = ROOT / "starter/results" / f"retrieval_dev_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    protocol = {
        "status": "preliminary_pending_source_review",
        "benchmark_sha256": hashlib.sha256(raw).hexdigest(),
        "corpus_sha256": hashlib.sha256(
            (DATA / "chunks.json").read_bytes()
        ).hexdigest(),
        "split": "dev",
        "filters": "disabled consistently for all methods",
        "context_budget_unit": "characters including source headers",
        "context_policy": "rank order, deduplicate, skip blocks that do not fit",
        "configs": CONFIGS,
        "hybrid": {"candidate_k": 10, "rrf_k": 60},
        "metrics": (
            "Gold quote match at source/page; q11 requires both IDs. "
            "Unanswerable cases excluded from retrieval denominators. "
            "Errors remain in answerable denominators as misses."
        ),
        "selection_rule": (
            "After source review and real-agent evaluation: prioritize "
            "grounded answer quality, then context evidence, then Hit@3, "
            "then lower measured cost and latency. Do not use test to tune."
        ),
        "limitations": (
            "No answer model, citation grading or abstention grading yet. "
            "Gold alternatives may be incomplete. Cached and uncached "
            "latencies must not be treated as equivalent."
        ),
    }

    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    retrievers = {
        "dense": DenseRetriever(),
        "tfidf": LexicalRetriever(),
        "hybrid": HybridRetriever(candidate_k=10, rrf_k=60),
    }

    summaries = []

    for config in CONFIGS:
        retriever = retrievers[config["method"]]
        records = []

        for question in questions:
            started = time.perf_counter()
            record = {
                "query_id": question["id"],
                "question": question["question"],
                "answerable": question["answerable"],
                "evidence_mode": question.get(
                    "evidence_mode", "single_chunk_alternatives"
                ),
            }

            try:
                # Only the question enters retrieval. Never pass gold.
                hits = retriever.search(
                    question["question"],
                    top_k=config["top_k"],
                    filters=None,
                )

                context, selected = pack_context(
                    hits, config["budget"]
                )

                usage = {}
                component_trace = None

                if config["method"] == "dense":
                    usage = dict(retriever.last_usage)
                elif config["method"] == "hybrid":
                    component_trace = retriever.last_trace
                    usage = dict(component_trace["embedding_usage"])

                record.update({
                    "status": "ok",
                    "results": hits,
                    "context": context,
                    "context_ids": [hit["id"] for hit in selected],
                    "context_characters": len(context),
                    "embedding_usage": usage,
                    "component_trace": component_trace,
                })

                if question["answerable"]:
                    record.update({
                        "hit_at_1": has_evidence(question, hits[:1]),
                        "hit_at_3": has_evidence(question, hits[:3]),
                        "context_evidence": has_evidence(
                            question, selected
                        ),
                    })

            except Exception as error:
                # Retain failure; do not replace it with mock results.
                record.update({
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                })

            record["elapsed_seconds"] = (
                time.perf_counter() - started
            )
            records.append(record)

            # Save after each query to preserve partial progress.
            trace_path = output_dir / f"{config['name']}_traces.json"
            trace_path.write_text(
                json.dumps(records, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            print(
                f"{config['name']} | {question['id']} "
                f"| {record['status']}"
            )

        answerable = [
            record for record in records if record["answerable"]
        ]
        single = [
            record for record in answerable
            if record["evidence_mode"] != "all_required_chunks"
        ]
        multi = [
            record for record in answerable
            if record["evidence_mode"] == "all_required_chunks"
        ]

        summary = {
            "config": config,
            "all_answerable": group_metrics(answerable),
            "single_chunk_cases": group_metrics(single),
            "multi_chunk_cases": group_metrics(multi),
            "errors": sum(r["status"] == "error" for r in records),
            "total_query_seconds": sum(
                r["elapsed_seconds"] for r in records
            ),
            "embedding_api_requests": sum(
                r.get("embedding_usage", {}).get("api_requests", 0)
                for r in records
            ),
            "embedding_tokens": sum(
                r.get("embedding_usage", {}).get("total_tokens", 0)
                for r in records
            ),
            "embedding_cache_hits": sum(
                bool(r.get("embedding_usage", {}).get("cache_hit"))
                for r in records
            ),
        }
        summaries.append(summary)

        (output_dir / "summary.json").write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print("\nPRELIMINARY DEV RESULTS")
    for summary in summaries:
        metrics = summary["all_answerable"]
        columns = []
        for key in ("hit_at_1", "hit_at_3", "context_evidence"):
            metric = metrics[key]
            columns.append(f"{key}={metric['n']}/{metric['N']}")

        print(
            summary["config"]["name"] + " | "
            + " | ".join(columns)
            + f" | errors={summary['errors']}"
        )

    print(f"\nSaved: {output_dir}")
    print("Source review and real-agent evaluation are still required.")


if __name__ == "__main__":
    main()