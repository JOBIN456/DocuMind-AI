# from qdrant_client import QdrantClient

# VECTOR_SIZE = 384

# qdrant_client = QdrantClient(host="localhost", port=6333)

# def collection_exists(name: str) -> bool:
#     names = [c.name for c in qdrant_client.get_collections().collections]
#     return name in names

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    SparseVectorParams,
    Distance
)

VECTOR_SIZE = 384

qdrant_client = QdrantClient(host="localhost", port=6333)


def collection_exists(name: str) -> bool:
    collections = qdrant_client.get_collections().collections
    return any(c.name == name for c in collections)


def setup_qdrant():
    if not collection_exists(COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams()
            }
        )

    return qdrant_client