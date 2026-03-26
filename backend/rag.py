from __future__ import annotations

import base64
import json
import os
from urllib import error, request

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()


class GeminiEmbedder:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2-preview")

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
            query_vector = self.embedder.embed_text(query, task_type="RETRIEVAL_QUERY")
            client = QdrantClient(url=self.url, api_key=self.api_key)
            response = client.query_points(
                collection_name=self.collection,
                query=query_vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            points = _extract_points(response)

            results: list[str] = []
            for point in points:
                payload = getattr(point, "payload", None) or {}
                text = payload.get("text") or payload.get("content") or payload.get("chunk")
                if isinstance(text, str) and text.strip():
                    results.append(text.strip())

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


def _extract_points(response: object) -> list[object]:
    points = getattr(response, "points", None)
    if isinstance(points, list):
        return points

    if isinstance(response, list):
        return response

    return []


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
