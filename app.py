"""Streamlit front end: upload a regulation, ask a question, see cited answers.

Run with:  streamlit run app.py
"""
import hashlib
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from pdf_loader import load_documents
from ragpipeline import build_rag, split_documents
from vectorstore import store_embeddings
from providers import missing_key

load_dotenv()
st.set_page_config(page_title="Regulatory RAG", page_icon="📑", layout="centered")
st.title("Regulatory RAG Analyzer")
st.caption("Ask a question about a regulation. Every answer cites the page and quote it relies on, "
           "and each quote is checked against the document.")

if missing_key():
    st.error(f"Set {missing_key()} in your environment or a .env file.")
    st.stop()


@st.cache_resource(show_spinner="Reading and indexing the regulation...")
def index_pdf(file_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    try:
        docs = load_documents(path)
        return build_rag(store_embeddings(split_documents(docs))), len(docs)
    finally:
        os.unlink(path)


pdf = st.file_uploader("Regulation PDF", type="pdf")
if pdf:
    data = pdf.getvalue()
    ask, n_pages = index_pdf(data)
    st.success(f"Indexed {pdf.name} ({n_pages} pages).")

    question = st.text_input("Your question", placeholder="What must a manufacturer do before placing a product on the market?")
    if question:
        with st.spinner("Searching the regulation..."):
            result = ask(question)
        st.subheader("Answer")
        st.markdown(result["answer"])
        st.subheader("Sources")
        if not result["citations"]:
            st.info("No supporting passages were cited.")
        for c in result["citations"]:
            badge = "✅ quote verified in document" if c["verified"] else "⚠️ quote not found verbatim, check manually"
            st.markdown(f"**[{c['source_id']}] Page {c['page']}** · {badge}")
            st.markdown(f"> {c['quote']}")
