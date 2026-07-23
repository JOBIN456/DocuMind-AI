import os
import hashlib
import sys
from typing import TypedDict, List, Any

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HOME"] = r"D:\newthings\StructRAG\hf_cache"

# goes up from api/ingest.py  →  StructRAG/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from langgraph.graph import StateGraph, START, END
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from sentence_transformers import SentenceTransformer
from db.qdrant_setup import qdrant_client
from qdrant_client.models import PointStruct, VectorParams, Distance,SparseVector
from fastembed import SparseTextEmbedding


# ── Load once at import time ──────────────────────────────────────
embed_model   = SentenceTransformer("all-MiniLM-L6-v2")
converter     = DocumentConverter()

sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

# ── State ─────────────────────────────────────────────────────────
class PdfState(TypedDict):
    docling_doc: Any
    chunks:      List[dict]
    embeddings:  List[List[float]]
    collection_name: str 


# ── Nodes ─────────────────────────────────────────────────────────
def chunk_node(state: PdfState):
    chunker = HybridChunker(
        tokenizer="BAAI/bge-small-en-v1.5",
        max_tokens=512,
        merge_peers=True,
    )

    raw_chunks = list(chunker.chunk(state["docling_doc"]))

    chunks = []

    texts = []

    for raw in raw_chunks:
        text = raw.text.strip()
        if not text:
            continue

        meta = raw.meta
        headings = meta.headings if meta.headings else []

        page_start = None
        page_end = None

        if hasattr(meta, "doc_items") and meta.doc_items:
            pages = []

            for item in meta.doc_items:
                if hasattr(item, "prov") and item.prov:
                    for prov in item.prov:
                        if hasattr(prov, "page_no"):
                            pages.append(prov.page_no)

            if pages:
                page_start = min(pages)
                page_end = max(pages)

        chunk = {
            "id": hashlib.md5(text.encode()).hexdigest()[:8],
            "text": text,
            "heading": " > ".join(headings) if headings else None,
            "page_start": page_start,
            "page_end": page_end,
        }

        chunks.append(chunk)
        texts.append(text)

    # Generate sparse vectors
    sparse_vectors = list(sparse_model.embed(texts))

    for chunk, sparse in zip(chunks, sparse_vectors):
        chunk["sparse_indices"] = sparse.indices.tolist()
        chunk["sparse_values"] = sparse.values.tolist()

    print(f"Docling produced {len(chunks)} chunks")

    return {"chunks": chunks}

def embed_node(state: PdfState):
    chunks = state["chunks"]
    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts)

    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()

    return {
        "chunks": chunks,
        "embeddings": embeddings.tolist()
    }

def qdrant_node(state: PdfState):
    points = []

    for i, chunk in enumerate(state["chunks"]):
        points.append(
            PointStruct(
                id=i,
                vector={
                    "dense": chunk["embedding"],
                    "sparse": SparseVector(
                        indices=chunk["sparse_indices"],
                        values=chunk["sparse_values"],
                    ),
                },
                payload={
                    "text": chunk["text"],
                    "heading": chunk.get("heading"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                },
            )
        )

    qdrant_client.upsert(
        collection_name=state["collection_name"],
        points=points,
    )

    print(f"✅ {len(points)} chunks inserted into → {state['collection_name']}")

    return {"chunks": state["chunks"]}

# ── Build graph once ──────────────────────────────────────────────
def _build_pipeline():
    graph = StateGraph(PdfState)
    graph.add_node("chunk",  chunk_node)
    graph.add_node("embed",  embed_node)
    graph.add_node("qdrant", qdrant_node)
    graph.add_edge(START,    "chunk")
    graph.add_edge("chunk",  "embed")
    graph.add_edge("embed",  "qdrant")
    graph.add_edge("qdrant", END)
    return graph.compile()

pipeline = _build_pipeline()


# ── Public function the router calls ─────────────────────────────
def run_ingest(pdf_path: str,collection_name: str) -> int:
    """
    Convert the PDF at pdf_path, embed it, and upsert into Qdrant.
    Returns the number of chunks inserted.
    """
    docling_doc = converter.convert(pdf_path).document

    result = pipeline.invoke({
        "docling_doc": docling_doc,
        "chunks":      [],
        "embeddings":  [],
        "collection_name": collection_name, 
    })

    return len(result["chunks"])