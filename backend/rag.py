from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()


class Retriever:
    def __init__(self):
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection = os.getenv("QDRANT_COLLECTION", "knowledge_base")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    def search(self, query: str, limit: int = 3) -> list[str]:
        docs = self._search_qdrant_langchain(query, limit)
        if docs:
            return docs

        return self._search_local(query, limit)

    def _search_qdrant_langchain(self, query: str, limit: int) -> list[str]:
        if not (self.url and self.collection):
            return []

        try:
            embeddings = OpenAIEmbeddings(model=self.embedding_model)
            vector_store = QdrantVectorStore.from_existing_collection(
                embedding=embeddings,
                url=self.url,
                api_key=self.api_key,
                collection_name=self.collection,
            )
            docs = vector_store.similarity_search(query, k=limit)
            return [doc.page_content for doc in docs if doc.page_content]
        except Exception:
            return []

    def _search_local(self, query: str, limit: int) -> list[str]:
        local_docs = [
            "Market cap is stock price multiplied by shares outstanding.",
            "A trailing P/E ratio compares current price to past earnings.",
            "Free cash flow is operating cash flow minus capital expenditures.",
            "Diversification can reduce unsystematic risk in a portfolio.",
        ]

        query_terms = set(query.lower().split())
        scored: list[tuple[int, str]] = []

        for doc in local_docs:
            score = len(query_terms.intersection(set(doc.lower().split())))
            scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [text for _, text in scored[:limit]]
