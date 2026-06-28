"""Loads a PDF file into LangChain documents."""
from langchain_community.document_loaders import PyPDFLoader


def load_documents(pdf_path):
    """pdf_path is a filesystem path to a PDF."""
    return PyPDFLoader(pdf_path).load()
