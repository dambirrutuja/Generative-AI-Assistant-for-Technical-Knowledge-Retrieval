from __future__ import annotations
import os, json, argparse
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .utils import load_config, ensure_dir, read_jsonl

DOCSTORE_FILE = "docstore.json"
INDEX_FILE = "faiss.index"

def _load_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)

def build_faiss_index(config_path: str, chunks_jsonl: str, index_dir: str) -> None:
    cfg = load_config(config_path)
    ecfg = cfg["embeddings"]
    model_name = ecfg["model_name"]
    batch_size = int(ecfg.get("batch_size", 64))
    normalize = bool(ecfg.get("normalize", True))

    ensure_dir(index_dir)
    chunks = read_jsonl(chunks_jsonl)
    chunks = [c for c in chunks if c.get("text")]

    texts = [c["text"] for c in chunks]
    meta = [{"chunk_id": c["chunk_id"], "source": c["source"], "meta": c.get("meta", {})} for c in chunks]

    embedder = _load_embedder(model_name)
    emb = embedder.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

    if normalize:
        faiss.normalize_L2(emb)

    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine if normalized
    index.add(emb.astype(np.float32))

    faiss.write_index(index, os.path.join(index_dir, INDEX_FILE))
    with open(os.path.join(index_dir, DOCSTORE_FILE), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_index(config_path: str, index_dir: str):
    cfg = load_config(config_path)
    ecfg = cfg["embeddings"]
    model_name = ecfg["model_name"]
    normalize = bool(ecfg.get("normalize", True))

    embedder = _load_embedder(model_name)
    index = faiss.read_index(os.path.join(index_dir, INDEX_FILE))
    with open(os.path.join(index_dir, DOCSTORE_FILE), "r", encoding="utf-8") as f:
        docstore = json.load(f)
    return embedder, normalize, index, docstore, cfg

def retrieve(embedder, normalize: bool, index, docstore: List[Dict[str, Any]], query: str, top_k: int) -> List[Dict[str, Any]]:
    q = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    if normalize:
        faiss.normalize_L2(q)
    scores, ids = index.search(q, top_k)
    out = []
    for s, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx < 0 or idx >= len(docstore):
            continue
        item = dict(docstore[idx])
        item["score"] = float(s)
        out.append(item)
    return out
