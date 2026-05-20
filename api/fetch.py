import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from db.qdrant_setup import qdrant_client  

load_dotenv()

# ── Load once ─────────────────────────────────────────────────────
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ── State ─────────────────────────────────────────────────────────
class RAGState(TypedDict):
    question:            str
    collection_name:     str        # ← added
    query_vector:        List[float]
    retrieved_chunks:    List[dict]
    relevance_scores:    List[str]
    rewrite_count:       int
    prompt:              str
    answer:              str
    hallucination_check: str


# ── Nodes ─────────────────────────────────────────────────────────
def embed_node(state: RAGState):
    print(f"\n🔍 Embedding question: {state['question']}")
    vector = embed_model.encode(state["question"]).tolist()
    return {"query_vector": vector}


def retrieve_node(state: RAGState):
    results = qdrant_client.query_points(          # ← shared client
        collection_name=state["collection_name"],  # ← dynamic collection
        query=state["query_vector"],
        limit=2,
        with_payload=True,
    ).points

    chunks = []
    for hit in results:
        chunks.append({
            "text":       hit.payload.get("text"),
            "heading":    hit.payload.get("heading"),
            "page_start": hit.payload.get("page_start"),
            "score":      round(hit.score, 4),
        })

    print(f"\n📄 Retrieved {len(chunks)} chunks from [{state['collection_name']}]:")
    for i, c in enumerate(chunks, 1):
        print(f"  {i}. [score={c['score']}] "
              f"{c['heading'] or 'No heading'} | "
              f"Page {c['page_start']} | "
              f"{c['text'][:80]}...")

    return {"retrieved_chunks": chunks}


def grade_docs_node(state: RAGState):
    scores = []
    for chunk in state["retrieved_chunks"]:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": f"""Is this document relevant to the question?
Question: {state['question']}
Document: {chunk['text'][:400]}
Answer ONLY 'relevant' or 'irrelevant'."""
            }],
            temperature=0,
        )
        scores.append(response.choices[0].message.content.strip().lower())

    print(f"📊 Relevance scores: {scores}")
    return {"relevance_scores": scores}


def rewrite_node(state: RAGState):
    if state["rewrite_count"] >= 2:
        print("⚠️ Max rewrites reached")
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
    new_q   = response.choices[0].message.content.strip()
    new_vec = embed_model.encode(new_q).tolist()
    print(f"🔄 Rewritten: {new_q}")
    return {
        "question":     new_q,
        "query_vector": new_vec,
        "rewrite_count": state["rewrite_count"] + 1,
    }


def prompt_node(state: RAGState):
    context_parts = []
    for i, chunk in enumerate(state["retrieved_chunks"], 1):
        heading = f"[{chunk['heading']}]" if chunk["heading"] else ""
        page    = f"(Page {chunk['page_start']})" if chunk["page_start"] else ""
        context_parts.append(f"Chunk {i} {heading} {page}:\n{chunk['text']}")

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
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You are a document assistant. Answer only from the provided context."
            },
            {
                "role": "user",
                "content": state["prompt"]
            }
        ],
        temperature=0.2,
        top_p=1,
        reasoning_effort="medium",
        stream=True,
        stop=None,
    )

    full_answer = ""
    for chunk in completion:
        token = chunk.choices[0].delta.content or ""
        print(token, end="", flush=True)
        full_answer += token

    print("\n")
    return {"answer": full_answer}


def hallucination_check_node(state: RAGState):
    context = "\n".join(c["text"][:300] for c in state["retrieved_chunks"])
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


# ── Routing ───────────────────────────────────────────────────────
def route_after_grading(state: RAGState):
    if "relevant" in state["relevance_scores"]:
        return "prompt"
    if state["rewrite_count"] >= 2:
        return "no_answer"
    return "rewrite"


def route_after_hallucination(state: RAGState):
    if state["hallucination_check"] == "grounded":
        return END
    if state["rewrite_count"] >= 2:
        return END
    return "prompt"


# ── Build pipeline once ───────────────────────────────────────────
def _build_rag():
    graph = StateGraph(RAGState)

    graph.add_node("embed",     embed_node)
    graph.add_node("retrieve",  retrieve_node)
    graph.add_node("grade",     grade_docs_node)
    graph.add_node("rewrite",   rewrite_node)
    graph.add_node("prompt",    prompt_node)
    graph.add_node("llm",       llm_node)
    graph.add_node("hallcheck", hallucination_check_node)
    graph.add_node("no_answer", no_answer_node)

    graph.add_edge(START,      "embed")
    graph.add_edge("embed",    "retrieve")
    graph.add_edge("retrieve", "grade")

    graph.add_conditional_edges("grade", route_after_grading, {
        "prompt":    "prompt",
        "rewrite":   "rewrite",
        "no_answer": "no_answer",
    })

    graph.add_edge("rewrite", "embed")
    graph.add_edge("prompt",  "llm")
    graph.add_edge("llm",     "hallcheck")

    graph.add_conditional_edges("hallcheck", route_after_hallucination, {
        "prompt": "prompt",
        END:      END,
    })

    graph.add_edge("no_answer", END)
    return graph.compile()

rag_pipeline = _build_rag()


# ── Public function the router calls ─────────────────────────────
def run_rag(question: str, collection_name: str) -> str:
    result = rag_pipeline.invoke({
        "question":            question,
        "collection_name":     collection_name,  # ← passed in
        "query_vector":        [],
        "retrieved_chunks":    [],
        "relevance_scores":    [],
        "rewrite_count":       0,
        "prompt":              "",
        "answer":              "",
        "hallucination_check": "",
    })
    return result["answer"]