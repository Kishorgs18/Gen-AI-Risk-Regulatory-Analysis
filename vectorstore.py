"""Embeds documents and stores them in an in-memory Chroma vector store."""
from langchain_chroma import Chroma

from providers import get_embeddings


def store_embeddings(docs):
    return Chroma.from_documents(docs, get_embeddings())
