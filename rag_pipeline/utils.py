from __future__ import annotations
import os, json, hashlib
from dataclasses import dataclass
from typing import Dict, Any, List, Iterable, Optional
import yaml

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

@dataclass
class Chunk:
    id: str
    text: str
    source: str
    meta: Dict[str, Any]

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
