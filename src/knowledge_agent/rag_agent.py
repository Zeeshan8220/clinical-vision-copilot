"""
Knowledge/RAG Agent -- retrieval-augmented question answering.
"""

import os
import chromadb
from groq import Groq

client_groq = Groq(api_key=os.environ.get("GROQ_API_KEY", "").strip())
CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
client_chroma = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client_chroma.get_or_create_collection(name="medical_knowledge")


def retrieve(question, n_results=3):
    """Step 1: find the most relevant knowledge snippets for this question."""
    results = collection.query(query_texts=[question], n_results=n_results)
    retrieved = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        retrieved.append({"text": doc, "source": meta["source"], "relevance_distance": dist})
    return retrieved


def answer_question(question, n_results=3):
    """Step 2: use retrieved snippets as context for a grounded LLM answer."""
    retrieved = retrieve(question, n_results=n_results)

    context = "\n\n".join(
        f"[Source: {r['source']}]\n{r['text']}" for r in retrieved
    )

    prompt = f"""Answer the user's question using ONLY the information in
the CONTEXT below. If the context does not contain enough information
to answer, say so explicitly rather than guessing from general
knowledge. Always cite which source(s) you used.

CONTEXT:
{context}

QUESTION: {question}

Answer concisely, then list the source(s) you used."""

    response = client_groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=400,
    )

    return {
        "answer": response.choices[0].message.content.strip(),
        "retrieved_sources": [r["source"] for r in retrieved],
    }


if __name__ == "__main__":
    result = answer_question("What does a flat ST slope on ECG suggest?")
    print("Answer:", result["answer"])
    print("\nSources used:", result["retrieved_sources"])
