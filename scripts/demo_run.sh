#!/usr/bin/env bash
set -e

python -m rag_pipeline.ingest --input_dir data/raw --out_dir data/processed --index_dir data/index
python -m rag_pipeline.evaluate --questions data/eval/questions.jsonl --index_dir data/index --out results/eval.json
echo "Run UI: streamlit run app/streamlit_app.py"
