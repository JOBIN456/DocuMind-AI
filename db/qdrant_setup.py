from qdrant_client import QdrantClient

VECTOR_SIZE = 384

qdrant_client = QdrantClient(host="localhost", port=6333)

def collection_exists(name: str) -> bool:
    names = [c.name for c in qdrant_client.get_collections().collections]
    return name in names