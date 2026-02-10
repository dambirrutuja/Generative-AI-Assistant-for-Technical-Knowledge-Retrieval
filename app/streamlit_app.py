import os
import sys
import subprocess
import streamlit as st

from rag_pipeline.rag import answer_question

st.set_page_config(page_title="Tech Knowledge RAG", layout="wide")
st.title("🔎 Generative AI Assistant for Technical Knowledge Retrieval")

# ---------------- Option A: Upload docs via UI + Rebuild Index ----------------
UPLOAD_DIR = "data/raw/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

with st.sidebar:
    st.header("Upload documents")

    uploaded_files = st.file_uploader(
        "Upload files (pdf, txt, md, csv, json, html)",
        accept_multiple_files=True,
        type=["pdf", "txt", "md", "csv", "json", "html", "htm"]
    )

    if uploaded_files:
        saved = 0
        for uf in uploaded_files:
            save_path = os.path.join(UPLOAD_DIR, uf.name)
            with open(save_path, "wb") as f:
                f.write(uf.getbuffer())
            saved += 1
        st.success(f"Saved {saved} file(s) to {UPLOAD_DIR}")

    if st.button("Rebuild Index"):
        with st.spinner("Rebuilding index (can take time for many PDFs)..."):
            result = subprocess.run(
                [sys.executable, "-m", "rag_pipeline.ingest",
                 "--input_dir", "data/raw",
                 "--out_dir", "data/processed",
                 "--index_dir", "data/index"],
                capture_output=True,
                text=True
            )

        if result.returncode == 0:
            st.success("Index rebuilt successfully ✅")
            if result.stdout.strip():
                st.code(result.stdout)
        else:
            st.error("Index rebuild failed ❌")
            st.code(result.stderr or "Unknown error")

    st.divider()

    st.header("Settings")
    config_path = st.text_input("Config path", "config/config.yaml")
    index_dir = st.text_input("Index dir", "data/index")
    processed_chunks = st.text_input("Processed chunks", "data/processed/chunks.jsonl")
    top_k = st.slider("Top K", 1, 10, 4)
# -----------------------------------------------------------------------------


st.write("Ask a question grounded in your document collection.")
q = st.text_input("Question", "What is the default API rate limit?")

if st.button("Ask"):
    with st.spinner("Retrieving + generating..."):
        res = answer_question(
            q,
            config_path=config_path,
            index_dir=index_dir,
            processed_chunks_path=processed_chunks,
            top_k=int(top_k),
        )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Answer")
        st.write(res["answer"])

    with col2:
        st.subheader("Latency")
        st.json(res["latency"])

    st.subheader("Retrieved chunks")
    for r in res["retrieved"]:
        st.markdown(
            f"- **{r['source']}** (score={r['score']:.3f}) — chunk `{r['chunk_id']}`"
        )
