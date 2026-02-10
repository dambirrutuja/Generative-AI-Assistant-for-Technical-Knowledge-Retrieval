from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline.rag import answer_question

app = FastAPI(title="Tech Knowledge RAG API")

class AskReq(BaseModel):
    question: str
    top_k: int = 4
    config_path: str = "config/config.yaml"
    index_dir: str = "data/index"
    processed_chunks_path: str = "data/processed/chunks.jsonl"

@app.post("/ask")
def ask(req: AskReq):
    return answer_question(
        req.question,
        config_path=req.config_path,
        index_dir=req.index_dir,
        processed_chunks_path=req.processed_chunks_path,
        top_k=req.top_k,
    )
