# Gen-AI Regulatory Risk Analyzer

A lightweight **RAG (Retrieval-Augmented Generation) microservice** that reads a
regulatory or compliance PDF and returns an LLM-generated summary of its **key
obligations**, along with the **risk KPIs** it references. Built as a FastAPI
service so it can drop straight into a compliance/risk workflow.

## What it does

```
PDF upload ─► PyPDFLoader ─► chunk + embed (OpenAI) ─► Chroma vector store
        ─► RetrievalQA (LangChain + OpenAI) ─► obligations summary
        ─► KPI extractor ─► detected risk metrics
```

`POST /analyze` with a PDF returns:

```json
{
  "summary": "Key obligations under this regulation are ...",
  "kpis": ["capital ratio", "liquidity coverage"]
}
```

## How it works

| Module | Role |
|---|---|
| `pdf_loader.py` | Loads & parses the regulatory PDF (`PyPDFLoader`) |
| `vectorstore.py` | Embeds chunks with `OpenAIEmbeddings`, stores in **Chroma** |
| `ragpipeline.py` | Builds a `RetrievalQA` chain (top-k retrieval + OpenAI LLM) |
| `kpi_extractor.py` | Scans the summary for risk KPIs (capital ratio, solvency margin, liquidity coverage, risk-weighted assets) |
| `main.py` | FastAPI app exposing `POST /analyze` |

## Tech stack

Python · **FastAPI** · **LangChain** · **Chroma** (vector DB) · OpenAI
(embeddings + LLM) · pdfplumber/PyPDF · Uvicorn

## Run it

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...        # required for embeddings + LLM
uvicorn main:app --reload          # docs at http://localhost:8000/docs

# then:
curl -F "pdf=@your_regulation.pdf" http://localhost:8000/analyze
```

## Why it's useful

Compliance teams spend hours reading dense regulatory texts to find what they're
*obliged* to do and which risk metrics apply. This service turns that into a
single API call — retrieval keeps the LLM grounded in the actual document
(reducing hallucination), and the KPI extractor flags the quantitative
thresholds that matter for risk reporting.

## Roadmap

- Swap regex KPI matching for an LLM extraction step with structured output.
- Add citations (page/section) to each summarised obligation.
- Containerise (a `Dockerfile` is included) and add a small Streamlit front-end.
