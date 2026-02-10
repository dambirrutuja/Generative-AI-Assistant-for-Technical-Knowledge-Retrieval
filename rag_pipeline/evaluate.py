from __future__ import annotations
import os, json, argparse, time, statistics
from typing import Dict, Any, List
from .rag import answer_question
from .utils import read_jsonl

def citation_coverage(answer: str, retrieved: List[Dict[str, Any]]) -> float:
    # simple signal: how many retrieved chunks are cited
    if not retrieved:
        return 0.0
    cited = 0
    for r in retrieved:
        token = f"[source:{r['source']}#{r['chunk_id']}]"
        if token in answer:
            cited += 1
    return cited / max(1, len(retrieved))

def retrieval_overlap(answer: str, retrieved: List[Dict[str, Any]]) -> float:
    # very lightweight "consistency": does answer share words with retrieved sources?
    import re
    words = set(re.findall(r"[a-zA-Z]{4,}", answer.lower()))
    if not words:
        return 0.0
    src_tokens = set()
    for r in retrieved:
        src_tokens.update(re.findall(r"[a-zA-Z]{4,}", (r.get("source","")).lower()))
    # This is intentionally weak; extend with real text overlap if desired.
    return len(words & src_tokens) / len(words)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--questions", required=True, help="JSONL with {id, question}")
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--processed_chunks", default="data/processed/chunks.jsonl")
    ap.add_argument("--out", default="results/eval.json")
    ap.add_argument("--top_k", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    qs = read_jsonl(args.questions)
    outputs = []
    latencies = []
    coverages = []
    for q in qs:
        res = answer_question(
            q["question"],
            config_path=args.config,
            index_dir=args.index_dir,
            processed_chunks_path=args.processed_chunks,
            top_k=args.top_k,
        )
        latencies.append(res["latency"]["total_s"])
        coverages.append(citation_coverage(res["answer"], res["retrieved"]))
        outputs.append({"id": q.get("id"), **res})

    report = {
        "n": len(outputs),
        "latency_s": {
            "mean": round(statistics.mean(latencies), 3) if latencies else None,
            "p50": round(statistics.median(latencies), 3) if latencies else None,
            "p90": round(statistics.quantiles(latencies, n=10)[8], 3) if len(latencies) >= 10 else None,
        },
        "citation_coverage": {
            "mean": round(statistics.mean(coverages), 3) if coverages else None,
        },
        "results": outputs,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote evaluation report to {args.out}")

if __name__ == "__main__":
    main()
