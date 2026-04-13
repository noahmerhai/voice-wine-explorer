import os
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic

load_dotenv()

CHROMA_DIR = "./chroma_db"
WINE_CSV_PATH = os.getenv("WINE_CSV_PATH", "./wines.csv")

SYSTEM_PROMPT = (
    "You are a knowledgeable sommelier assistant for a wine retailer. You will be given a set of wine records "
    "retrieved from an inventory database, followed by a user's question. Answer using ONLY the wine data provided. "
    "Do not invent facts, prices, ratings, vintages, or any details not present in the retrieved records. "
    "If the data does not support the question, say so clearly and briefly. "
    "Guidelines: Keep answers conversational, warm, and concise (2-4 sentences unless listing wines). "
    "When listing wines, be selective — surface the most relevant options, not every match. "
    "For recommendation questions (gifts, occasions, food pairings), reason from the data you have — "
    "color, region, varietal, price, and ratings. "
    "If a question is ambiguous, give your best answer from the available data and note any assumptions. "
    "Never say based on my training or reference anything outside the retrieved records.\n\n"
    "Wine Data:\n{context}"
)


def _flatten_row(row: pd.Series) -> str:
    parts = []
    for col, val in row.items():
        if pd.isna(val) or val == "":
            continue
        if col == "professional_ratings":
            try:
                ratings = json.loads(val) if isinstance(val, str) else val
                if isinstance(ratings, list):
                    for r in ratings:
                        source = r.get("source", "")
                        score = r.get("score", "")
                        note = r.get("note", "")
                        parts.append(f"{source} {score}/100: {note}")
                    continue
            except (json.JSONDecodeError, TypeError):
                pass
        parts.append(f"{col}: {val}")
    return " | ".join(parts)


def _load_documents() -> list[Document]:
    df = pd.read_csv(WINE_CSV_PATH)
    docs = []
    for _, row in df.iterrows():
        text = _flatten_row(row)
        if text.strip():
            docs.append(Document(page_content=text))
    return docs


def build_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    if Path(CHROMA_DIR).exists():
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
        if vectorstore._collection.count() == 0:
            documents = _load_documents()
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=CHROMA_DIR,
            )
    else:
        documents = _load_documents()
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0,
        timeout=30,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
