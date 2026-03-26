# TESTING QDRANT RETRIEVAL

from backend.rag import Retriever
print(Retriever().search('What is Jodoa properties?', limit=3))


# WIPING QDRANT MEMORY

# from qdrant_client import QdrantClient
# client = QdrantClient(url="http://localhost:6333")
# client.delete_collection(collection_name="knowledge_base")
# print("Cleared Qdrant memory!")
