"""Chooses the chat model and embeddings from environment variables.

Defaults: OpenAI for both when OPENAI_API_KEY is set; otherwise Groq for chat
(free tier) and local FastEmbed embeddings that run on CPU with no API key.

Override with LLM_PROVIDER=openai|groq and EMBEDDINGS=openai|local.
"""
import os


def llm_provider():
    return os.getenv("LLM_PROVIDER") or ("openai" if os.getenv("OPENAI_API_KEY") else "groq")


def embeddings_provider():
    return os.getenv("EMBEDDINGS") or ("openai" if os.getenv("OPENAI_API_KEY") else "local")


def get_chat_model():
    if llm_provider() == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), temperature=0)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)


def get_embeddings():
    if embeddings_provider() == "local":
        from langchain_community.embeddings import FastEmbedEmbeddings
        return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings()


def missing_key():
    """Name of the API key the current configuration needs but lacks, or None."""
    if llm_provider() == "groq" and not os.getenv("GROQ_API_KEY"):
        return "GROQ_API_KEY"
    if "openai" in (llm_provider(), embeddings_provider()) and not os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY"
    return None
