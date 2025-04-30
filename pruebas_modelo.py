import os
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import json
import faiss
from sklearn.preprocessing import normalize
# Elige el modelo adecuado
model_name = 'all-MiniLM-L6-v2' # o 'paraphrase-multilingual-MiniLM-L12-v2'
embedding_model = SentenceTransformer(model_name)

df = pd.read_csv("customer-support-tickets-en-derfinitivo.csv")

# Asegúrate de que 'problem_text' existe y no tiene nulos (usa .fillna('') si es necesario)
texts_to_embed = df['problem_text'].tolist()

# Nombres de los archivos
index_filename = "support_cases_faiss.index"
metadata_filename = "support_cases_metadata.json"

# Verificar si los archivos ya existen
if os.path.exists(index_filename) and os.path.exists(metadata_filename):
    # Cargar Índice FAISS
    index = faiss.read_index(index_filename)
    print(f"Índice FAISS cargado desde {index_filename}")

    # Cargar Metadatos
    with open(metadata_filename, 'r', encoding='utf-8') as f:
        metadata_store = json.load(f)
    print(f"Metadatos cargados desde {metadata_filename}")
else:
    # Generar el índice y los metadatos si no existen
    print("Archivos no encontrados. Generando nuevos índices y metadatos...")

    # Generar embeddings
    embeddings = embedding_model.encode(texts_to_embed, show_progress_bar=True)
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    # Normalizar embeddings
    faiss.normalize_L2(embeddings)

    # Crear el índice FAISS
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    # Crear los metadatos
    metadata_store = []
    for i in range(len(df)):
        row = df.iloc[i]
        metadata_store.append({
            'id': i,
            'subject': row['subject'],
            'body': row['body'],
            'answer': row['answer'],
            'tags': row['tags'],
        })

# Ahora, si FAISS te devuelve el índice 'k', puedes obtener sus datos con metadata_store[k]
print(f"Metadatos almacenados para {len(metadata_store)} casos.")

def search_similar_cases(query_text, k=5):
    # 1. Generar embedding para la consulta 
    query_embedding = embedding_model.encode([query_text])[0].astype(np.float32)

    # 2. Normalizar
    # O usando FAISS:
    query_embedding_norm = np.copy(query_embedding) # Copiar para no modificar el original
    faiss.normalize_L2(query_embedding_norm.reshape(1, -1)) # Normaliza in-place

    # 3. Reshape para la búsqueda (FAISS espera un array 2D: n_queries x dimension)
    query_vector = query_embedding_norm.reshape(1, -1)

    # 4. Realizar la búsqueda en FAISS
    # index.search devuelve Distances (D) e Indices (I)
    distances, indices = index.search(query_vector, k)

    # 5. Recuperar y formatear resultados
    results = []
    for i in range(k):
        result_index = indices[0][i] # Índice del vecino más cercano número i
        similarity_score = distances[0][i] # Puntuación (Producto Interno en este caso)

        # Ignorar resultados si el índice es -1 (puede pasar con algunos índices)
        if result_index == -1:
            continue

        # Obtener metadatos
        result_metadata = metadata_store[result_index]

        results.append({
            'id': result_metadata['id'],
            'score': similarity_score,
            'subject': result_metadata['subject'],
            'body': result_metadata['body'],
            'answer': result_metadata['answer'],
            'tags': result_metadata['tags']
        })
    return results

# --- Probar la búsqueda ---
query = "I'm having problems with the encryption of my data"
search_results = search_similar_cases(query, k=5)

print(f"\nResultados para la consulta: '{query}'")
for result in search_results:
    print(f"\n--- ID: {result['id']} (Score: {result['score']:.4f}) ---")
    print(f"  Subject: {result['subject']}")
    print(f"  Tags: {result['tags']}")
    print(f"  Body: {result['body']}")
    print(f"  Answer: {result['answer']}")

# --- GUARDAR ---
# 1. Guardar Índice FAISS
index_filename = "support_cases_faiss.index"
faiss.write_index(index, index_filename)
print(f"Índice FAISS guardado en {index_filename}")

# 2. Guardar Metadatos (como JSON)
metadata_filename = "support_cases_metadata.json"
with open(metadata_filename, 'w', encoding='utf-8') as f:
    json.dump(metadata_store, f, ensure_ascii=False, indent=4) # indent=4 para que sea legible
print(f"Metadatos guardados en {metadata_filename}")