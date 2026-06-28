"""Embeds documents and stores them in an in-memory Chroma vector store."""
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def store_embeddings(docs):
    return Chroma.from_documents(docs, OpenAIEmbeddings())
