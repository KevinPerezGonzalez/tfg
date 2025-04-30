from flask import Flask, render_template, request
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)

# Cargar modelo y datos
model_name = 'all-MiniLM-L6-v2'
embedding_model = SentenceTransformer(model_name)

index = faiss.read_index("support_cases_faiss.index")
with open("support_cases_metadata.json", "r", encoding="utf-8") as f:
    metadata_store = json.load(f)

def search_similar_cases(query_text, k=5):
    query_embedding = embedding_model.encode([query_text])[0].astype(np.float32)
    query_embedding_norm = np.copy(query_embedding)
    faiss.normalize_L2(query_embedding_norm.reshape(1, -1))
    query_vector = query_embedding_norm.reshape(1, -1)

    distances, indices = index.search(query_vector, k)
    results = []
    for i in range(k):
        idx = indices[0][i]
        score = distances[0][i]
        if idx == -1:
            continue
        meta = metadata_store[idx]
        results.append({
            'id': meta['id'],
            'score': round(score, 4),
            'subject': meta['subject'],
            'body': meta['body'],
            'answer': meta['answer'],
            'tags': meta['tags']
        })
    return results

@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    query = ""
    if request.method == "POST":
        query = request.form["query"]
        results = search_similar_cases(query, k=5)
    return render_template("index.html", query=query, results=results)

if __name__ == "__main__":
    app.run(debug=True)
