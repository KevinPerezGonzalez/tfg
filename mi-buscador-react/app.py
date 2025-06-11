# app.py (Flask Backend)
import os
import faiss
import json
import numpy as np
import pandas as pd # Necesario para la función load_and_prepare_data
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from flask_cors import CORS # Para permitir peticiones desde el frontend de React

# --- CONFIGURACIÓN ---
# Elige UN modelo para que esta API lo sirva.
# El índice FAISS y los metadatos deben corresponder a este modelo.
CHOSEN_MODEL_NAME = 'sentence-t5-base' # Modelo por defecto para la API
SANITIZED_MODEL_NAME = CHOSEN_MODEL_NAME.replace('/', '_').replace('-', '_')
INDEX_FILENAME = f"support_cases_faiss_{SANITIZED_MODEL_NAME}.index"
METADATA_FILENAME_GLOBAL = "support_cases_metadata_GLOBAL.json"
SOURCE_DATA_FILE = "customer-support-tickets-en-derfinitivo.csv" # Necesario si los metadatos no existen

app = Flask(__name__)
CORS(app) # Habilita CORS para todas las rutas

# --- Variables Globales para los recursos cargados ---
embedding_model_global = None
index_global = None
metadata_store_global = None

# --- Función de Ayuda para Cargar Datos (similar a la anterior) ---
def load_and_prepare_data_for_api(source_file, metadata_file_path):
    current_df = None
    current_metadata_store = None
    
    if os.path.exists(metadata_file_path):
        print(f"Cargando metadatos globales desde {metadata_file_path}...")
        with open(metadata_file_path, 'r', encoding='utf-8') as f:
            current_metadata_store = json.load(f)
        print(f"Metadatos cargados: {len(current_metadata_store)} casos.")
        # Si necesitas df para alguna otra cosa, cárgalo. Aquí solo para crear metadatos si no existen.
    else:
        print(f"Archivo de metadatos {metadata_file_path} no encontrado. Creándolo desde {source_file}...")
        try:
            current_df = pd.read_csv(source_file)
            if 'problem_text' not in current_df.columns:
                if 'subject' in current_df.columns and 'body' in current_df.columns:
                    current_df['problem_text'] = current_df['subject'].fillna('') + ' ' + current_df['body'].fillna('')
                else:
                    raise ValueError("Columnas 'subject'/'body' no encontradas para crear 'problem_text'")

            if 'tags' in current_df.columns and current_df['tags'].notna().any() and isinstance(current_df['tags'].dropna().iloc[0], str):
                current_df['tags'] = current_df['tags'].fillna('').apply(
                    lambda x: [tag.strip() for tag in x.split(',') if tag.strip()] if isinstance(x, str) else []
                )
            elif 'tags' not in current_df.columns:
                current_df['tags'] = [[] for _ in range(len(current_df))]
            
            current_metadata_store = []
            for i in range(len(current_df)):
                row = current_df.iloc[i]
                current_metadata_store.append({
                    'doc_id_internal': i,
                    'original_ticket_id': row.get('Ticket ID', row.get('id', i)),
                    'subject': row.get('subject', 'N/A'),
                    'body': row.get('body', 'N/A'),
                    'answer': row.get('answer', 'N/A'),
                    'tags': row.get('tags', [])
                })
            
            with open(metadata_file_path, 'w', encoding='utf-8') as f:
                json.dump(current_metadata_store, f, ensure_ascii=False, indent=4)
            print(f"Metadatos globales guardados en {metadata_file_path}.")

        except FileNotFoundError:
            print(f"Error CRÍTICO: No se encontró el archivo de datos fuente: {source_file}")
            return None
        except Exception as e:
            print(f"Error CRÍTICO al procesar DataFrame: {e}")
            return None
            
    return current_metadata_store

# --- CARGAR RECURSOS UNA SOLA VEZ AL INICIO ---
def load_resources():
    global embedding_model_global, index_global, metadata_store_global
    print("Iniciando carga de recursos para la API...")
    try:
        print(f"Cargando modelo: {CHOSEN_MODEL_NAME}...")
        embedding_model_global = SentenceTransformer(CHOSEN_MODEL_NAME)
        
        print(f"Cargando metadatos...")
        metadata_store_global = load_and_prepare_data_for_api(SOURCE_DATA_FILE, METADATA_FILENAME_GLOBAL)
        if not metadata_store_global:
            raise FileNotFoundError("No se pudieron cargar o crear los metadatos.")

        print(f"Cargando índice FAISS: {INDEX_FILENAME}...")
        if not os.path.exists(INDEX_FILENAME):
            raise FileNotFoundError(f"Archivo de índice FAISS {INDEX_FILENAME} no encontrado. Asegúrate de generarlo primero con el script de comparación/indexación para el modelo {CHOSEN_MODEL_NAME}.")
        index_global = faiss.read_index(INDEX_FILENAME)
        
        # Verificación de consistencia simple
        if index_global.ntotal != len(metadata_store_global):
            print(f"Advertencia: El número de vectores en el índice ({index_global.ntotal}) no coincide con los metadatos ({len(metadata_store_global)}).")
        if index_global.d != embedding_model_global.get_sentence_embedding_dimension():
            print(f"Advertencia: Dimensión del índice cargado ({index_global.d}) no coincide con la del modelo ({embedding_model_global.get_sentence_embedding_dimension()}).")

        print(f"Recursos cargados: Modelo, Índice ({index_global.ntotal} vectores), Metadatos ({len(metadata_store_global)} casos)")
    except Exception as e:
        print(f"Error crítico al cargar recursos globales: {e}")
        # La aplicación podría no funcionar correctamente si los recursos no se cargan.
        embedding_model_global = None
        index_global = None
        metadata_store_global = None

# --- FUNCIÓN DE BÚSQUEDA (Reutilizada y adaptada) ---
def search_similar_cases_api(query_text, k_results=5):
    if not embedding_model_global or not index_global or not metadata_store_global:
        print("Error: Recursos (modelo, índice o metadatos) no cargados en la API.")
        return []
    try:
        query_embedding = embedding_model_global.encode([query_text])[0].astype(np.float32)
        query_embedding_norm = np.copy(query_embedding)
        faiss.normalize_L2(query_embedding_norm.reshape(1, -1))
        query_vector = query_embedding_norm.reshape(1, -1)
        
        distances, indices = index_global.search(query_vector, k_results)
        
        results = []
        if indices.size == 0 or (len(indices[0]) > 0 and indices[0][0] == -1) : return results

        for i in range(min(k_results, len(indices[0]))):
            result_index = indices[0][i]
            if result_index == -1: continue
            if 0 <= result_index < len(metadata_store_global):
                similarity_score = distances[0][i]
                retrieved_metadata = metadata_store_global[result_index]
                results.append({
                    'doc_id_internal': retrieved_metadata.get('doc_id_internal', result_index),
                    'original_ticket_id': retrieved_metadata.get('original_ticket_id', 'N/A'),
                    'score': float(similarity_score), # Asegurar que es float para JSON
                    'subject': retrieved_metadata.get('subject', 'N/A'),
                    'body': retrieved_metadata.get('body', 'N/A'),
                    'answer': retrieved_metadata.get('answer', 'N/A'),
                    'tags': retrieved_metadata.get('tags', [])
                })
            else:
                 print(f"Advertencia: Índice FAISS {result_index} fuera de rango para metadatos.")
        return results
    except Exception as e:
        print(f"Error durante la búsqueda en la API: {e}")
        return []

# --- RUTA DE LA API ---
@app.route('/api/search', methods=['GET'])
def api_search():
    query = request.args.get('query', '')
    if not query:
        return jsonify({'error': 'Query parameter is missing'}), 400
    
    if not embedding_model_global or not index_global or not metadata_store_global:
        return jsonify({'error': 'Backend resources not loaded. Please check server logs.'}), 503

    print(f"API: Recibida consulta: '{query}'")
    results = search_similar_cases_api(query, k_results=5)
    print(f"API: Devolviendo {len(results)} resultados.")
    return jsonify(results)

@app.route('/api/health', methods=['GET'])
def health_check():
    if embedding_model_global and index_global and metadata_store_global:
        return jsonify({
            "status": "OK",
            "model_loaded": CHOSEN_MODEL_NAME,
            "index_vectors": index_global.ntotal if index_global else 0,
            "metadata_entries": len(metadata_store_global) if metadata_store_global else 0
        }), 200
    else:
        return jsonify({"status": "ERROR", "message": "Recursos no cargados"}), 503
        
# --- CARGAR RECURSOS AL INICIAR LA APP ---
load_resources()

# --- EJECUTAR LA APLICACIÓN ---
if __name__ == '__main__':
    # NO usar debug=True en producción final, pero útil para desarrollo
    app.run(debug=True, host='0.0.0.0', port=5000)