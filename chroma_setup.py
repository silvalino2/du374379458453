import numpy as np
import pickle
import os
import csv
from fastembed import TextEmbedding

model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
store = []

STORE_FILE = "store.pkl"

def save_store():
    with open(STORE_FILE, "wb") as f:
        pickle.dump(store, f)

def embed(text: str):
    return list(model.embed([text]))[0]

def add_fact(id: str, text: str, source: str = "", country: str = "Nigeria"):
    embedding = embed(text)
    store.append({"id": id, "text": text, "source": source, "country": country, "embedding": embedding})

def rebuild_from_corpus():
    global store
    store = []
    with open("corpus.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            combined_text = f"Q: {row['question_or_title']}\nA: {row['answer_or_content']}"
            add_fact(
                id=str(idx),
                text=combined_text,
                source=row.get("source_url", ""),
                country="Nigeria"
            )
    save_store()

def load_store():
    global store
    if os.path.exists(STORE_FILE):
        with open(STORE_FILE, "rb") as f:
            store = pickle.load(f)
        if store and len(store[0]["embedding"]) != len(embed("test")):
            rebuild_from_corpus()
    else:
        rebuild_from_corpus()

load_store()

def query_facts(question: str, n: int = 3):
    q_embedding = embed(question)
    scored = []
    for item in store:
        sim = np.dot(q_embedding, item["embedding"]) / (
            np.linalg.norm(q_embedding) * np.linalg.norm(item["embedding"])
        )
        scored.append((sim, item["text"], item["source"]))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:n]
