"""FastAPI entrypoint for the regulatory RAG analyzer.

POST a regulatory PDF to /analyze; returns a summary of key obligations with
verified citations (page + quote), plus the risk KPIs detected in it.

Uses OpenAI if OPENAI_API_KEY is set, otherwise Groq plus local embeddings (see providers.py).
"""
import os
import shutil
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, Form, UploadFile

from ragpipeline import build_rag, split_documents
from pdf_loader import load_documents
from vectorstore import store_embeddings
from kpi_extractor import extract_kpis
from providers import embeddings_provider, llm_provider, missing_key

load_dotenv()
app = FastAPI(title="Regulatory Risk RAG Analyzer")

DEFAULT_QUESTION = "Summarize key obligations under this regulation."


@app.get("/health")
def health():
    return {"status": "ok", "llm": llm_provider(), "embeddings": embeddings_provider(), "missing_key": missing_key()}


@app.post("/analyze")
async def analyze(pdf: UploadFile, question: str = Form(DEFAULT_QUESTION)):
    # PyPDFLoader needs a file path, so persist the upload to a temp file first.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(pdf.file, tmp)
        tmp_path = tmp.name
    try:
        chunks = split_documents(load_documents(tmp_path))
        ask = build_rag(store_embeddings(chunks))
        result = ask(question)
    finally:
        os.unlink(tmp_path)
    return {
        "summary": result["answer"],
        "citations": result["citations"],
        "kpis": extract_kpis(result["answer"]),
    }
