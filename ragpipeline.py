"""Builds a Retrieval-Augmented QA chain using LangChain Expression Language.

Uses only stable langchain-core primitives, so it runs on current LangChain
(v1) without depending on the removed `langchain.chains.RetrievalQA`.
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

_PROMPT = ChatPromptTemplate.from_template(
    "You are a regulatory-compliance assistant. Using only the context below, "
    "answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"
)


def _format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def build_rag(vector_db):
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    return (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | _PROMPT
        | llm
        | StrOutputParser()
    )
