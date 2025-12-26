import json
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from ollama import chat  # pip install ollama

CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

MODEL_NAME = config["model_name"]
TOP_K = int(config["top_k"])

DB_DIR = Path(__file__).parent / "faiss_db"
INDEX_PATH = DB_DIR / "repo.index"
DOCS_PATH = DB_DIR / "docs.json"

index = faiss.read_index(str(INDEX_PATH))

with open(DOCS_PATH, "r", encoding="utf-8") as f:
    docs = json.load(f)

embed_model = SentenceTransformer(r"C:\all-MiniLM-L6-v2")  # loading embeddings


def retrieve_chunks(query: str, top_k: int = TOP_K):
    q_emb = embed_model.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype="float32")

    scores, idxs = index.search(q_emb, top_k)

    results = []
    for i, score in zip(idxs[0].tolist(), scores[0].tolist()):
        if i < 0 or i >= len(docs):
            continue
        item = docs[i]
        results.append((item["text"], {"path": item["path"], "chunk_id": item["chunk_id"], "score": score}))
    return results


def build_prompt(question: str, chunks_with_meta):
    context_parts = []
    for chunk, meta in chunks_with_meta:
        path = meta.get("path", "unknown")
        context_parts.append(f"File: {path}\n{chunk}")

    context_text = "\n\n---\n\n".join(context_parts)

    system_msg = (
        "You are a coding assistant for this repository. "
        "Answer the user's question using ONLY the provided code snippets. "
        "If the answer is not clearly in the snippets, say you don't know. "
        "Be concise and, when relevant, mention file names."
    )

    user_msg = (
        f"Repository context:\n{context_text}\n\n"
        f"Question: {question}"
    )

    return system_msg, user_msg


def answer_question(question: str) -> str:
    print("[assistant] Retrieving chunks...")
    chunks = retrieve_chunks(question, TOP_K)
    print(f"[assistant] Retrieved {len(chunks)} chunks")

    if not chunks:
        return "I couldn't find any relevant code in the index."

    system_msg, user_msg = build_prompt(question, chunks)
    print("[assistant] Calling model...")

    response = chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    print("[assistant] Got response from model.")
    return response["message"]["content"]


def main():
    print("Repo assistant (FAISS) ready. Type your question (or 'exit'):")
    while True:
        q = input("> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        ans = answer_question(q)
        print("\n--- Answer ---")
        print(ans)
        print("--------------\n")


if __name__ == "__main__":
    main()
