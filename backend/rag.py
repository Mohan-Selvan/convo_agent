from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()


def get_embeddings_model() -> GoogleGenerativeAIEmbeddings:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) is required.")

    model = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
    return GoogleGenerativeAIEmbeddings(
        model=model,
        google_api_key=api_key,
    )


class Retriever:
    def __init__(self):
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection = os.getenv("QDRANT_COLLECTION", "knowledge_base")
        self.embedder = get_embeddings_model()

    def search(self, query: str, limit: int = 4) -> list[str]:
        if not (self.url and self.collection):
            return []

        try:
            vector_store = QdrantVectorStore.from_existing_collection(
                collection_name=self.collection,
                embedding=self.embedder,
                url=self.url,
                api_key=self.api_key,
            )
            docs = vector_store.similarity_search(query, k=limit)
            return [
                doc.page_content
                for doc in docs
                if isinstance(doc.page_content, str) and doc.page_content
            ]
        except Exception:
            return []
