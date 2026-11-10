"""Retrieval-augmented QA over a regulation, with verified citations.

Every answer comes back with the passages it relies on: the PDF page number and
a short quote. Quotes are checked against the retrieved text, so a citation the
model invents (or paraphrases) is flagged instead of being shown as evidence.
"""
import re
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from providers import get_chat_model, llm_provider

_PROMPT = ChatPromptTemplate.from_template(
    "You are a regulatory-compliance assistant. Answer the question using ONLY "
    "the numbered sources below. After each claim, cite the source number in "
    "square brackets, e.g. [2]. For every source you cite, copy a short quote "
    "(one sentence or less) word for word from that source as evidence. If the "
    "sources do not answer the question, say so plainly.\n\n"
    "Sources:\n{context}\n\nQuestion: {question}"
)


class Citation(BaseModel):
    source_id: int = Field(description="Number of the cited source, e.g. 2")
    quote: str = Field(description="Short quote copied word for word from that source")


class CitedAnswer(BaseModel):
    answer: str = Field(description="Answer with [n] citations after each claim")
    citations: List[Citation]


def split_documents(docs, chunk_size=1200, chunk_overlap=150):
    """Split pages into chunks while keeping each chunk's page number."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def _normalise(text):
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"-\s*\n\s*", "", text)  # words hyphenated across lines
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_is_verified(quote, source_text):
    """True only if the quote really appears in the source text.

    A quote shortened with an ellipsis passes only if every fragment appears
    verbatim and in the same order, so trimming is allowed but rewording is not.
    """
    source = _normalise(source_text)
    fragments = [f.strip(" .,;:\"'") for f in re.split(r"\.\.\.|\u2026", _normalise(quote))]
    fragments = [f for f in fragments if f]
    if not fragments:
        return False
    position = 0
    for fragment in fragments:
        found = source.find(fragment, position)
        if found == -1:
            return False
        position = found + len(fragment)
    return True


def _format_sources(docs):
    return "\n\n".join(
        f"[{i}] (page {d.metadata.get('page', 0) + 1})\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


def build_rag(vector_db, k=5):
    """Return a function question -> {answer, citations} with verified quotes."""
    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    # Groq's gpt-oss models are more reliable with JSON schema mode than tool calls.
    method = "json_schema" if llm_provider() == "groq" else "function_calling"
    llm = get_chat_model().with_structured_output(CitedAnswer, method=method)

    def ask(question):
        docs = retriever.invoke(question)
        result = (_PROMPT | llm).invoke({"context": _format_sources(docs), "question": question})
        citations = []
        for c in result.citations:
            if not 1 <= c.source_id <= len(docs):
                continue  # model cited a source that was never retrieved
            doc = docs[c.source_id - 1]
            citations.append({
                "source_id": c.source_id,
                "page": doc.metadata.get("page", 0) + 1,
                "quote": c.quote,
                "verified": quote_is_verified(c.quote, doc.page_content),
            })
        return {"answer": result.answer, "citations": citations}

    return ask
