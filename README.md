# Gen-AI Regulatory Risk Analyzer

A **RAG (Retrieval-Augmented Generation) service** that reads a regulatory or
compliance PDF and answers questions about it, such as the key obligations or
what applies to a product, **with verified citations**. Every answer comes back
with the page number and a short quote it relies on, and each quote is checked
against the document so invented or reworded "evidence" is flagged.

Runs as a FastAPI service and a small Streamlit app.

## What it does

```
PDF ─► PyPDFLoader ─► chunk (page numbers kept) ─► embed ─► Chroma vector store
    ─► retrieve top-k numbered sources ─► LLM answers with [n] citations
       and a word-for-word quote per source (structured output)
    ─► quote verification against the retrieved text ─► answer + citations
```

Example (EU General Product Safety Regulation, 2023/988):

> **Q:** What must a manufacturer do before placing a product on the market?
> **A:** ...keep the technical documentation up to date [3] ... ensure products bear a
> type, batch or serial number [3] ... accompany the product with clear instructions
> and safety information [4] ...
>
> [3] Page 24 · ✅ verified · "Manufacturers shall ensure that the technical documentation
> referred to in paragraph 2 is up to date."

If the document does not answer the question, it says so instead of guessing.

## How verification works

The model must copy a short quote word for word from each source it cites.
`quote_is_verified()` normalises whitespace, quotes and line-break hyphenation,
then checks the quote really appears in that source. A quote shortened with
"..." passes only if every fragment appears verbatim and in order, so trimming is
allowed but rewording is not. Unverified quotes are shown with a warning, never
as evidence. Unit tests live in `tests/`.

## How it works

| Module | Role |
|---|---|
| `pdf_loader.py` | Loads the regulatory PDF (`PyPDFLoader`), one document per page |
| `ragpipeline.py` | Chunks pages, retrieves numbered sources, asks for a cited answer, verifies quotes |
| `vectorstore.py` | Embeds chunks and stores them in **Chroma** |
| `providers.py` | Picks the chat model and embeddings (OpenAI, or Groq plus local FastEmbed) |
| `kpi_extractor.py` | Flags risk KPIs mentioned in an answer (capital ratio, liquidity coverage, ...) |
| `main.py` | FastAPI app exposing `POST /analyze` |
| `app.py` | Streamlit UI: upload a PDF, ask a question, see cited answers |

## Tech stack

Python · **FastAPI** · **LangChain** · **Chroma** · OpenAI or **Groq** (gpt-oss-120b) ·
**FastEmbed** local embeddings (BAAI/bge-small-en-v1.5) · Streamlit · pytest

## Run it

```bash
pip install -r requirements.txt

# Option A: OpenAI for embeddings and chat
export OPENAI_API_KEY=sk-...
# Option B: free tier, Groq for chat + local embeddings (no OpenAI key needed)
export GROQ_API_KEY=gsk_...

streamlit run app.py               # UI at http://localhost:8501
uvicorn main:app --reload          # API docs at http://localhost:8000/docs

curl -F "pdf=@your_regulation.pdf" \
     -F "question=What obligations apply to importers?" \
     http://localhost:8000/analyze

pytest -q                          # verification tests
```

`POST /analyze` returns:

```json
{
  "summary": "Answer with [n] citations ...",
  "citations": [{"source_id": 3, "page": 24, "quote": "...", "verified": true}],
  "kpis": ["capital ratio"]
}
```

## Why it's useful

Compliance teams spend hours reading dense regulatory texts to find what they
are obliged to do. A summary alone is not enough, because in compliance a
confident wrong answer is worse than no answer. This service grounds every
answer in the document and shows checkable evidence, so a reviewer can confirm
each claim in seconds.

## Roadmap

- Section and article numbers in citations, alongside page numbers.
- Replace regex KPI matching with structured LLM extraction.
- Evaluation set of questions with expected pages to track retrieval accuracy.
