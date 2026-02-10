from __future__ import annotations
import os, time
from typing import Dict, Any, List, Tuple

from .retriever import load_index, retrieve
from .llm import build_llm

def _format_context(chunks: List[Dict[str, Any]], processed_chunks_path: str) -> str:
    # Load chunk texts from processed chunks.jsonl (lightweight docstore holds only metadata)
    # We keep this simple: read once per call. For production, cache in memory.
    import json
    texts = {}
    with open(processed_chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts[obj["chunk_id"]] = obj.get("text", "")

    parts = []
    for c in chunks:
        cid = c["chunk_id"]
        src = c["source"]
        txt = texts.get(cid, "")
        parts.append(f"[source:{src}#{cid}]\n{txt}")
    return "\n\n---\n\n".join(parts)

def answer_question(
    question: str,
    config_path: str = "config/config.yaml",
    index_dir: str = "data/index",
    processed_chunks_path: str = "data/processed/chunks.jsonl",
    top_k: int = 4,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    embedder, normalize, index, docstore, cfg = load_index(config_path, index_dir)
    retrieved = retrieve(embedder, normalize, index, docstore, question, top_k=top_k)
    t_retr = time.perf_counter()

    context = _format_context(retrieved, processed_chunks_path)
    prompt = cfg["prompt"]["template"].format(context=context, question=question)
    llm = build_llm(cfg)
    answer = llm.generate(prompt)
    t_gen = time.perf_counter()

    return {
        "question": question,
        "answer": answer,
        "retrieved": retrieved,
        "latency": {
            "total_s": round(t_gen - t0, 3),
            "retrieve_s": round(t_retr - t0, 3),
            "generate_s": round(t_gen - t_retr, 3),
        },
    }
