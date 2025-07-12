"""
RAG helper that:
  • connects to the Weaviate container on the Docker network
  • pulls embeddings from OpenAI
  • exposes ask_question(query) → { result, source_documents }
"""
from pathlib import Path
import os
from dotenv import load_dotenv
from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain.chains import RetrievalQA
from typing import Iterator
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# ────────────────────────────────
# 1.  ENV & connection constants
# ────────────────────────────────
# Loads .env (root of the project in this setup)
load_dotenv(Path(__file__).parents[1] / "infra/.env")

WEAVIATE_HTTP_URL   = os.getenv("WEAVIATE_URL")
WEAVIATE_GRPC_PORT  = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
COLLECTION          = os.getenv("WEAVIATE_COLLECTION",   "InvestopediaArticles")
TEXT_KEY            = os.getenv("WEAVIATE_TEXT_KEY",     "text")
print("✅ Using Weaviate URL:", WEAVIATE_HTTP_URL)
print("🔐 OpenAI key loaded:", os.getenv("OPENAI_API_KEY"))

# ────────────────────────────────
# 2.  Weaviate client (v4)
# ────────────────────────────────
client = WeaviateClient(
    connection_params=ConnectionParams.from_url(
        url        = WEAVIATE_HTTP_URL,   # http://weaviate:8080 inside Docker
        grpc_port  = WEAVIATE_GRPC_PORT,  # 50051
    )
)
client.connect()          # <- mandatory in v4 (raises if it can’t)

# ────────────────────────────────
# 3.  VectorStore + Retriever
# ────────────────────────────────
embeddings = OpenAIEmbeddings()           # uses OPENAI_API_KEY from env
vector_db  = WeaviateVectorStore(
    client     = client,
    index_name = COLLECTION,
    text_key   = TEXT_KEY,
    embedding  = embeddings,
)
retriever = vector_db.as_retriever()

# ────────────────────────────────
# 4.  LLM + RetrievalQA chain
# ────────────────────────────────
llm       = ChatOpenAI(model="gpt-4o")   # pick any chat-completion model
rag_chain = RetrievalQA.from_chain_type(
    llm                     = llm,
    retriever               = retriever,
    return_source_documents = True,
)

# ────────────────────────────────
# 5.  Public helper
# ────────────────────────────────
def ask_question(query: str) -> dict:
    """
    Run a question through the RAG chain and return:
      { "result": <answer str>,
        "source_documents": [doc, …] }
    """
    try:
        return rag_chain.invoke(query)
    except Exception as exc:
        return {
            "result": f"Error: {exc}",
            "source_documents": [],
        }

def stream_question(query: str) -> Iterator[str]:
    """
    Generator that yields tokens from the streaming LLM as they are produced
    """
    prompt = PromptTemplate.from_template("Question: {question}")
    chain = LLMChain(llm=ChatOpenAI(streaming=True), prompt=prompt)

    for chunk in chain.stream({"question": query}):
        yield chunk["text"]  # 🔥 this must be a string
