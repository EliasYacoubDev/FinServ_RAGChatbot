# ingest/loader.py  ───────────────────────────────────────────────────────
"""
Read output/raw_pages.jl  →  strip article HTML  →  chunk  →  embed  →
upsert into Weaviate (v4).  Run this inside the app container once the
crawler has finished:
    docker exec -it infra-app-1 python ingest/loader.py
"""
from __future__ import annotations

import json, re, os
from pathlib import Path
from typing import List

import bs4
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore

from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams
from weaviate.classes import config as wvc

# ─────────────────────────────────────────────────────────────────────────
root_dir = Path(__file__).resolve().parents[1]   # FinServ_RAGChatbot/
env_path = root_dir / "infra" / ".env"           # FinServ_RAGChatbot/infra/.env
load_dotenv(env_path)                            # ← now OPENAI_API_KEY is set

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RAW_FILE   = Path("output/raw_pages.jl")
COLLECTION = "InvestopediaArticles"              # any valid class name
TEXT_KEY   = "text"
CHUNK_SIZE = 800
OVERLAP    = 100

# Weaviate is a *separate* container; DNS name “weaviate” works from app
client = WeaviateClient(
    connection_params=ConnectionParams.from_url(
        "http://localhost:8080",   # REST endpoint
        grpc_port=50051            # gRPC endpoint (mapped through Docker)
    )
)
client.connect()

def extract_article_text(html: str) -> str | None:
    soup     = bs4.BeautifulSoup(html, "lxml")
    article  = soup.find("article") or soup.select_one("#mntl-sc-page_1-0")
    if not article:
        return None
    text = re.sub(r"\s+", " ", article.get_text(" ", strip=True))
    return text

def load_and_chunk(path: Path) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP
    )
    docs: List[Document] = []

    for line in path.read_text().splitlines():
        rec  = json.loads(line)
        body = extract_article_text(rec["body"])
        if not body:
            continue
        for chunk in splitter.split_text(body):
            docs.append(Document(page_content=chunk,
                                 metadata={"url": rec["url"]}))
    return docs

def ensure_schema() -> None:
    if client.collections.exists(COLLECTION):
        return
    client.collections.create(
    name=COLLECTION,

    # ------- properties -------
    properties=[
        wvc.Property(
            name=TEXT_KEY,
            data_type=wvc.DataType.TEXT     # <- use enum, not string
        ),
        wvc.Property(
            name="source",
            data_type=wvc.DataType.TEXT
        ),
    ],

    # ------- module configs (vector + generative) -------
    vectorizer_config = wvc.Configure.Vectorizer.text2vec_openai(),
    generative_config = wvc.Configure.Generative.openai()
)
    print(f"🆕  Created collection {COLLECTION}")

def main() -> None:
    docs = load_and_chunk(RAW_FILE)
    if not docs:
        raise SystemExit("❌ No docs found in raw_pages.jl")

    ensure_schema()

    store = WeaviateVectorStore(
        client     = client,
        index_name = COLLECTION,
        text_key   = TEXT_KEY,
        embedding  = OpenAIEmbeddings(),           # uses OPENAI_API_KEY env
    )
    store.add_documents(docs)
    print(f"Upserted {len(docs)} text chunks into Weaviate")
    client.close()
if __name__ == "__main__":
    main()
