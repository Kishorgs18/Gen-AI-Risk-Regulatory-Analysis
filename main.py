"""FastAPI entrypoint for the regulatory RAG analyzer.

POST a regulatory PDF to /analyze; returns an LLM summary of key obligations
plus the risk KPIs detected in it.
"""
from fastapi import FastAPI, UploadFile

from ragpipeline import build_rag
from pdf_loader import load_documents
from vectorstore import store_embeddings
from kpi_extractor import extract_kpis

app = FastAPI()

@app.post("/analyze")
async def analyze(pdf: UploadFile):
    docs = load_documents(pdf.file)
    vector_db = store_embeddings(docs)
    rag = build_rag(vector_db)
    summary = rag.run("Summarize key obligations under this regulation.")
    kpis = extract_kpis(summary)
    return {"summary": summary, "kpis": kpis}