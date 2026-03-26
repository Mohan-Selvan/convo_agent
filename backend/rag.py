from __future__ import annotations

import base64
import json
import os
from urllib import error, request

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

_PDF_PREFIX = "__PDF_BASE64__::"


class GeminiEmbedder(Embeddings):
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2-preview")

    def embed_query(self, text: str) -> list[float]:
        return self.embed_text(text, task_type="RETRIEVAL_QUERY")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            if text.startswith(_PDF_PREFIX):
                b64_data = text[len(_PDF_PREFIX) :]
                pdf_bytes = base64.b64decode(b64_data)
                vectors.append(self.embed_pdf_bytes(pdf_bytes))
            else:
                vectors.append(self.embed_text(text, task_type="RETRIEVAL_DOCUMENT"))
        return vectors

    def embed_text(self, text: str, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
        payload = {
            "content": {
                "parts": [{"text": text}],
            },
            "taskType": task_type,
        }
        return self._embed(payload)

    def embed_pdf_bytes(self, pdf_bytes: bytes) -> list[float]:
        payload = {
            "content": {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "application/pdf",
                            "data": base64.b64encode(pdf_bytes).decode("utf-8"),
                        }
                    }
                ]
            },
            "taskType": "RETRIEVAL_DOCUMENT",
        }
        return self._embed(payload)

    def _embed(self, payload: dict) -> list[float]:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for embeddings.")

        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent"
            f"?key={self.api_key}"
        )

        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Gemini embedding request failed: {body}") from exc

        return _extract_vector(data)


class Retriever:
    def __init__(self):
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection = os.getenv("QDRANT_COLLECTION", "knowledge_base")
        self.embedder = GeminiEmbedder()

    def search(self, query: str, limit: int = 3) -> list[str]:
        docs = self._search_qdrant(query, limit)
        if docs:
            return docs

        return self._search_local(query, limit)

    def _search_qdrant(self, query: str, limit: int) -> list[str]:
        if not (self.url and self.collection and self.embedder.api_key):
            return []

        try:
            vector_store = QdrantVectorStore.from_existing_collection(
                collection_name=self.collection,
                embedding=self.embedder,
                url=self.url,
                api_key=self.api_key,
            )
            docs = vector_store.similarity_search(query, k=limit)

            results: list[str] = []
            for doc in docs:
                content = doc.page_content if isinstance(doc.page_content, str) else ""
                metadata = doc.metadata or {}

                if metadata.get("is_native_pdf_embedding"):
                    source = metadata.get("source", "unknown source")
                    results.append(f"[Relevant content from PDF: {source}]")
                    continue

                if content and not content.startswith(_PDF_PREFIX):
                    results.append(content)

            return results
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


def build_pdf_document(pdf_bytes: bytes, source: str) -> Document:
    encoded = base64.b64encode(pdf_bytes).decode("utf-8")
    return Document(
        page_content=f"{_PDF_PREFIX}{encoded}",
        metadata={
            "source": source,
            "type": "pdf",
            "is_native_pdf_embedding": True,
        },
    )


def _extract_vector(response_json: dict) -> list[float]:
    embedding = response_json.get("embedding")
    if isinstance(embedding, dict):
        values = embedding.get("values")
        if isinstance(values, list):
            return [float(v) for v in values]

    embeddings = response_json.get("embeddings")
    if isinstance(embeddings, list) and embeddings:
        first = embeddings[0]
        if isinstance(first, dict):
            values = first.get("values")
            if isinstance(values, list):
                return [float(v) for v in values]

    raise RuntimeError(f"Unexpected embedding response: {response_json}")
