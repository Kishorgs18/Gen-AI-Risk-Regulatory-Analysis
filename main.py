"""FastAPI entrypoint for the regulatory RAG analyzer.

POST a regulatory PDF to /analyze; returns an LLM summary of key obligations
plus the risk KPIs detected in it.

Requires OPENAI_API_KEY in the environment (embeddings + chat model).
"""
import os
import shutil
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile

from ragpipeline import build_rag
from pdf_loader import load_documents
from vectorstore import store_embeddings
from kpi_extractor import extract_kpis

load_dotenv()
app = FastAPI(title="Regulatory Risk RAG Analyzer")


@app.get("/health")
def health():
    return {"status": "ok", "openai_key_set": bool(os.getenv("OPENAI_API_KEY"))}


@app.post("/analyze")
async def analyze(pdf: UploadFile):
    # PyPDFLoader needs a file path, so persist the upload to a temp file first.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(pdf.file, tmp)
        tmp_path = tmp.name
    try:
        docs = load_documents(tmp_path)
        vector_db = store_embeddings(docs)
        rag = build_rag(vector_db)
        summary = rag.invoke("Summarize key obligations under this regulation.")
        kpis = extract_kpis(summary)
    finally:
        os.unlink(tmp_path)
    return {"summary": summary, "kpis": kpis}
