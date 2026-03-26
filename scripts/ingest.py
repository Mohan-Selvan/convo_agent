from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def ingest_directory(
    source_dir: Path,
    collection_name: str,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> int:
    files = sorted(source_dir.rglob("*.txt")) + sorted(source_dir.rglob("*.md"))
    if not files:
        return 0

    raw_docs: list[Document] = []
    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        if text.strip():
            raw_docs.append(
                Document(
                    page_content=text,
                    metadata={"source": str(file_path)},
                )
            )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(raw_docs)
    if not chunks:
        return 0

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    if not qdrant_url:
        raise ValueError("QDRANT_URL is required in .env for ingestion.")

    embeddings = OpenAIEmbeddings(model=embedding_model)
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name,
    )

    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest .txt/.md docs into Qdrant")
    parser.add_argument(
        "--source-dir",
        default="data",
        help="Directory containing .txt/.md files (default: data)",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION", "knowledge_base"),
        help="Qdrant collection name",
    )
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=100)
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
