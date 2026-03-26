from __future__ import annotations

import argparse
import sys
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Make project root importable when running: python scripts/ingest.py ...
sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.rag import GeminiEmbedder

load_dotenv()


def ingest_directory(
    source_dir: Path,
    collection_name: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> int:
    files = sorted(source_dir.rglob("*.pdf"))
    files += sorted(source_dir.rglob("*.txt"))
    files += sorted(source_dir.rglob("*.md"))

    if not files:
        return 0

    embedder = GeminiEmbedder()
    qdrant_url = _get_env("QDRANT_URL")
    qdrant_api_key = _get_optional_env("QDRANT_API_KEY")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    points: list[PointStruct] = []

    for file_path in files:
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            pdf_chunks = _extract_pdf_chunks(file_path, chunk_size, chunk_overlap)
            if pdf_chunks:
                for page_no, chunk in pdf_chunks:
                    vector = embedder.embed_text(chunk, task_type="RETRIEVAL_DOCUMENT")
                    payload = {
                        "source": str(file_path),
                        "type": "pdf",
                        "page": page_no,
                        "text": chunk,
                    }
                    points.append(
                        PointStruct(
                            id=str(uuid4()),
                            vector=vector,
                            payload=payload,
                        )
                    )
            else:
                pdf_bytes = file_path.read_bytes()
                vector = embedder.embed_pdf_bytes(pdf_bytes)
                payload = {
                    "source": str(file_path),
                    "type": "pdf",
                    "text": f"[PDF indexed natively with {embedder.model}]",
                }
                points.append(
                    PointStruct(
                        id=str(uuid4()),
                        vector=vector,
                        payload=payload,
                    )
                )
            continue

        text = file_path.read_text(encoding="utf-8", errors="ignore")
        for chunk in _chunk_text(text, chunk_size, chunk_overlap):
            vector = embedder.embed_text(chunk, task_type="RETRIEVAL_DOCUMENT")
            payload = {
                "source": str(file_path),
                "type": suffix.lstrip("."),
                "text": chunk,
            }
            points.append(
                PointStruct(
                    id=str(uuid4()),
                    vector=vector,
                    payload=payload,
                )
            )

    if not points:
        return 0

    vector_size = len(points[0].vector)
    _ensure_collection(client, collection_name, vector_size)

    batch_size = 64
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=collection_name,
            points=points[i : i + batch_size],
            wait=True,
        )

    return len(points)


def _ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    if client.collection_exists(collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def _extract_pdf_chunks(
    pdf_path: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    chunks: list[tuple[int, str]] = []

    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        for chunk in _chunk_text(text, chunk_size, chunk_overlap):
            chunks.append((i, chunk))

    return chunks


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)

    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def _get_env(name: str) -> str:
    import os

    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} is required in .env")
    return value


def _get_optional_env(name: str) -> str | None:
    import os

    return os.getenv(name)


def main() -> None:
    import os

    parser = argparse.ArgumentParser(description="Ingest .pdf/.txt/.md docs into Qdrant")
    parser.add_argument(
        "--source-dir",
        default="data",
        help="Directory containing .pdf/.txt/.md files (default: data)",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION", "knowledge_base"),
        help="Qdrant collection name",
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.exists() or not source_dir.is_dir():
        raise ValueError(f"Invalid source directory: {source_dir}")

    inserted = ingest_directory(
        source_dir=source_dir,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Ingested {inserted} chunks into collection '{args.collection}'.")


if __name__ == "__main__":
    main()
