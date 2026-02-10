from __future__ import annotations
import os, re, json, argparse
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from bs4 import BeautifulSoup
from pypdf import PdfReader
import pandas as pd

from .utils import load_config, ensure_dir, sha1_text, save_jsonl

TEXT_EXT = {".txt", ".md", ".markdown", ".log"}
HTML_EXT = {".html", ".htm"}
JSON_EXT = {".json"}
CSV_EXT  = {".csv"}

def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _read_pdf(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for p in reader.pages:
        try:
            parts.append(p.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n".join(parts)

def _read_html(path: str) -> str:
    html = _read_text_file(path)
    soup = BeautifulSoup(html, "html.parser")
    # drop scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text("\n")

def _read_json(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        obj = json.load(f)
    return json.dumps(obj, ensure_ascii=False, indent=2)

def _read_csv(path: str) -> str:
    df = pd.read_csv(path)
    return df.to_csv(index=False)

def load_document(path: str) -> str:
    ext = os.path.splitext(path.lower())[1]
    if ext in TEXT_EXT:
        return _read_text_file(path)
    if ext in HTML_EXT:
        return _read_html(path)
    if ext in JSON_EXT:
        return _read_json(path)
    if ext in CSV_EXT:
        return _read_csv(path)
    if ext == ".pdf":
        return _read_pdf(path)
    raise ValueError(f"Unsupported file type: {ext} ({path})")

def clean_text(text: str) -> str:
    # Normalize whitespace
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def chunk_text(text: str, chunk_size: int, overlap: int, min_chars: int) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end]
        if len(chunk) >= min_chars:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def iter_files(input_dir: str) -> List[str]:
    paths = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            p = os.path.join(root, fn)
            ext = os.path.splitext(fn.lower())[1]
            if ext in TEXT_EXT | HTML_EXT | JSON_EXT | CSV_EXT or ext == ".pdf":
                paths.append(p)
    return sorted(paths)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    icfg = cfg["ingest"]
    chunk_size = int(icfg["chunk_size"])
    overlap = int(icfg["chunk_overlap"])
    min_chars = int(icfg["min_chunk_chars"])
    max_docs = icfg.get("max_docs", None)
    workers = args.workers if args.workers is not None else int(icfg.get("workers", 1))

    ensure_dir(args.out_dir)
    ensure_dir(args.index_dir)

    files = iter_files(args.input_dir)
    if max_docs:
        files = files[: int(max_docs)]

    rows = []
    for path in tqdm(files, desc="Ingesting"):
        try:
            raw = load_document(path)
            text = clean_text(raw)
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap, min_chars=min_chars)
            rel = os.path.relpath(path, args.input_dir).replace("\\", "/")
            for i, c in enumerate(chunks):
                cid = f"{sha1_text(rel)}_{i:05d}"
                rows.append({
                    "chunk_id": cid,
                    "source": rel,
                    "text": c,
                    "meta": {"path": rel, "chunk_index": i, "chars": len(c)}
                })
        except Exception as e:
            rows.append({
                "chunk_id": f"ERR_{sha1_text(path)}",
                "source": os.path.relpath(path, args.input_dir).replace("\\", "/"),
                "text": "",
                "meta": {"error": str(e)}
            })

    chunks_path = os.path.join(args.out_dir, "chunks.jsonl")
    save_jsonl(chunks_path, rows)

    # Build index
    from .retriever import build_faiss_index
    build_faiss_index(config_path=args.config, chunks_jsonl=chunks_path, index_dir=args.index_dir)

    print(f"Saved chunks: {chunks_path}")
    print(f"Built index in: {args.index_dir}")

if __name__ == "__main__":
    main()
