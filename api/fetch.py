import os
import sys
import random
from collections import defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoders import CrossEncoder
from groq import Groq
from dotenv import load_dotenv
from db.qdrant_setup import qdrant_client  

load_dotenv()

# ── Load once ─────────────────────────────────────────────────────
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ── State ─────────────────────────────────────────────────────────
class RAGState(TypedDict):
    question: str
    collection_name: str
    query_vector: List[float]
    retrieved_chunks: List[dict]
    generation_attempts: int
    prompt: str
    answer: str
    hallucination_check: str
    dense_chunks: List[dict]
    sparse_chunks: List[dict]
    query_type: str
    rewrite_count: int


# ── Nodes ─────────────────────────────────────────────────────────

def query_classifier_node(state: RAGState):
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """
You are a query classifier.

Classify the user's question into exactly ONE category.

Categories:

retrieval
- Questions that require searching the document.
- Questions about facts in uploaded documents.
- Questions asking "what", "who", "when", "where", "explain", "summarize", etc.

greeting
- Hi
- Hello
- Good morning
- Good evening
- Thanks
- Bye

chit_chat
- How are you?
- Tell me a joke.
- Who created you?
- What's your favorite color?

unsupported
- Harmful requests
- Illegal requests
- Anything outside assistant capability.

Return ONLY one word:

retrieval
greeting
chit_chat
unsupported
"""
            },
            {
                "role": "user",
                "content": state["question"]
            }
        ]
    )

    query_type = response.choices[0].message.content.strip().lower()
    print(f"📌 Query Type: {query_type}")

    return {"query_type": query_type}


def route_after_classifier(state: RAGState):
    query_type = state.get("query_type", "").lower()

    if query_type == "retrieval":
        return "embed"
    elif query_type in ["greeting", "chit_chat"]:
        return "general_response_node"
    elif query_type == "unsupported":
        return "unsupported_node"
    
    return "general_response_node"


def unsupported_node(state: RAGState):
    return {"answer": "Sorry, I can't assist with that request."}


def general_response_node(state: RAGState):
    query_type = state["query_type"]
    question = state["question"].lower().strip()

    greetings = [
        "Hello! How can I help you today?",
        "Hi! What can I do for you?",
        "Hey! How may I assist you?",
        "Welcome! Feel free to ask me anything."
    ]

    thanks = [
        "You're welcome!",
        "Happy to help!",
        "Glad I could help!",
        "Anytime!"
    ]

    if query_type == "greeting":
        if any(word in question for word in ["thank", "thanks"]):
            answer = random.choice(thanks)
        else:
            answer = random.choice(greetings)
    elif query_type == "chit_chat":
        answer = (
            "I'm designed to answer questions related to the uploaded documents. "
            "Please ask a document-related question."
        )
    else:
        answer = "I'm unable to process that request."

    return {"answer": answer}


def embed_node(state: RAGState):
    print(f"\n🔍 Embedding question: {state['question']}")
    vector = embed_model.encode(state["question"]).tolist()
    return {"query_vector": vector}


def dense_retrieve_node(state: RAGState):
    print(f"\n🔍 Dense retrieval for: {state['question']}")

    results = qdrant_client.query_points(
        collection_name=state["collection_name"],
        query=state["query_vector"],
        limit=10,
        with_payload=True,
    ).points

    dense_chunks = []
    for hit in results:
        dense_chunks.append({
            "id": str(hit.id),
            "text": hit.payload.get("text"),
            "heading": hit.payload.get("heading"),
            "page_start": hit.payload.get("page_start"),
            "dense_score": float(hit.score),
        })

    return {"dense_chunks": dense_chunks}


def sparse_retrieve_node(state: RAGState):
    print(f"🔍 Sparse retrieval for: {state['question']}")
    
    # Using dense retrieval as fallback if sparse not available
    # If you have sparse retrieval setup, replace this section
    sparse_chunks = []
    
    return {"sparse_chunks": sparse_chunks}


def fusion_node(state: RAGState):
    """
    Weighted Reciprocal Rank Fusion (WRRF)
    Dense retrieval has higher importance than sparse retrieval.
    """

    DENSE_WEIGHT = 0.7
    SPARSE_WEIGHT = 0.3
    K = 60

    fused_scores = defaultdict(float)
    chunk_lookup = {}

    # Dense Retrieval Contribution
    for rank, chunk in enumerate(state["dense_chunks"]):
        score = DENSE_WEIGHT * (1 / (rank + K))
        fused_scores[chunk["id"]] += score
        chunk_lookup[chunk["id"]] = chunk

    # Sparse Retrieval Contribution
    for rank, chunk in enumerate(state["sparse_chunks"]):
        score = SPARSE_WEIGHT * (1 / (rank + K))
        fused_scores[chunk["id"]] += score
        if chunk["id"] not in chunk_lookup:
            chunk_lookup[chunk["id"]] = chunk

    # Sort by fused score
    ranked_ids = sorted(
        fused_scores.keys(),
        key=lambda x: fused_scores[x],
        reverse=True,
    )

    final_chunks = []
    for chunk_id in ranked_ids[:10]:
        chunk = chunk_lookup[chunk_id].copy()
        chunk["hybrid_score"] = round(fused_scores[chunk_id], 6)
        final_chunks.append(chunk)

    print("\n========== Hybrid Retrieval ==========\n")
    for i, chunk in enumerate(final_chunks, 1):
        print(
            f"{i}. "
            f"Score={chunk['hybrid_score']:.6f} | "
            f"Dense={chunk.get('dense_score', 0):.4f} | "
            f"Sparse={chunk.get('bm25_score', 0):.4f} | "
            f"{chunk.get('heading', 'No Heading')}"
        )
    print("\n======================================\n")

    return {"retrieved_chunks": final_chunks}


def rerank_node(state: RAGState, top_k: int = 10):
    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        return {"retrieved_chunks": []}

    pairs = [
        (state["question"], chunk["text"])
        for chunk in chunks
    ]

    scores = reranker.predict(
        pairs,
        batch_size=32,
        show_progress_bar=False
    )

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    reranked = sorted(
        chunks,
        key=lambda x: x["rerank_score"],
        reverse=True
    )[:top_k]

    return {"retrieved_chunks": reranked}


def rewrite_node(state: RAGState):
    if state["rewrite_count"] >= 2:
        print("⚠️ Max rewrites reached, using original")
        return {"rewrite_count": state["rewrite_count"]}

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"""Rewrite this question to improve document retrieval.
Original: {state['question']}
Return ONLY the rewritten question."""
        }],
        temperature=0.3,
    )
    new_q = response.choices[0].message.content.strip()
    new_vec = embed_model.encode(new_q).tolist()
    print(f"🔄 Rewritten: {new_q}")
    return {
        "question": new_q,
        "query_vector": new_vec,
        "rewrite_count": state["rewrite_count"] + 1
    }


def prompt_node(state: RAGState):
    context_parts = []

    for i, chunk in enumerate(state["retrieved_chunks"], 1):
        heading = f"[{chunk['heading']}]" if chunk.get("heading") else ""
        page = f"(Page {chunk['page_start']})" if chunk.get("page_start") else ""
        context_parts.append(
            f"Chunk {i} {heading} {page}:\n{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the document."

CONTEXT:
{context}

QUESTION:
{state['question']}

ANSWER:"""

    return {"prompt": prompt}


def llm_node(state: RAGState):
    print("\n🤖 Answer:\n")

    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a document assistant. Answer only from the provided context."},
            {"role": "user", "content": state["prompt"]}
        ],
        temperature=0.2,
        top_p=1,
        stream=True,
        stop=None
    )

    full_answer = ""
    for chunk in completion:
        token = chunk.choices[0].delta.content or ""
        print(token, end="", flush=True)
        full_answer += token

    print("\n")
    return {
        "answer": full_answer,
        "generation_attempts": state.get("generation_attempts", 0) + 1,
    }


def hallucination_check_node(state: RAGState):
    context = "\n".join(c["text"][:500] for c in state["retrieved_chunks"])
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"""Is this answer grounded in the context?
Context: {context}
Answer: {state['answer']}
Reply ONLY 'grounded' or 'hallucinated'."""
        }],
        temperature=0,
    )
    result = response.choices[0].message.content.strip().lower()
    print(f"🧠 Hallucination check: {result}")
    return {"hallucination_check": result}


def no_answer_node(state: RAGState):
    print("❌ Question is not relevant to this document.")
    return {"answer": "I don't know based on the document."}


def route_after_hallucination(state: RAGState):
    if state["hallucination_check"] == "grounded":
        return END
    if state.get("generation_attempts", 0) >= 2:
        return END
    return "prompt"


# ── Build pipeline ───────────────────────────────────────────────
def _build_rag():
    graph = StateGraph(RAGState)
    
    # Add nodes
    graph.add_node("query_classifier_node", query_classifier_node)
    graph.add_node("unsupported_node", unsupported_node)
    graph.add_node("general_response_node", general_response_node)
    graph.add_node("embed", embed_node)
    graph.add_node("dense_retrieve", dense_retrieve_node)
    graph.add_node("sparse_retrieve", sparse_retrieve_node)
    graph.add_node("fusion", fusion_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("prompt", prompt_node)
    graph.add_node("llm", llm_node)
    graph.add_node("hallcheck", hallucination_check_node)
    graph.add_node("no_answer", no_answer_node)

    # Main Flow
    graph.add_edge(START, "query_classifier_node")
    
    graph.add_conditional_edges(
        "query_classifier_node",
        route_after_classifier,
        {
            "embed": "embed",
            "general_response_node": "general_response_node",
            "unsupported_node": "unsupported_node",
        }
    )

    # Retrieval pipeline
    graph.add_edge("embed", "dense_retrieve")
    graph.add_edge("dense_retrieve", "sparse_retrieve")
    graph.add_edge("sparse_retrieve", "fusion")
    graph.add_edge("fusion", "rerank")
    graph.add_edge("rerank", "prompt")

    # Rewrite loop
    graph.add_edge("rewrite", "embed")

    # Generation
    graph.add_edge("prompt", "llm")
    graph.add_edge("llm", "hallcheck")

    # Hallucination routing
    graph.add_conditional_edges(
        "hallcheck",
        route_after_hallucination,
        {
            "prompt": "prompt",
            END: END,
        }
    )

    # End points
    graph.add_edge("unsupported_node", END)
    graph.add_edge("general_response_node", END)
    graph.add_edge("no_answer", END)

    return graph.compile()


rag_pipeline = _build_rag()


# ── Public function ─────────────────────────────────────────────
def run_rag(question: str, collection_name: str) -> str:
    result = rag_pipeline.invoke({
        "question": question,
        "collection_name": collection_name,
        "query_vector": [],
        "retrieved_chunks": [],
        "generation_attempts": 0,
        "prompt": "",
        "answer": "",
        "hallucination_check": "",
        "dense_chunks": [],
        "sparse_chunks": [],
        "query_type": "",
        "rewrite_count": 0,
    })
    return result["answer"]