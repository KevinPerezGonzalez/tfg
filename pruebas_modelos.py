import os
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import json
import faiss
# from sklearn.preprocessing import normalize # No se usa, FAISS tiene su propia normalización

# --- CONFIGURACIÓN GENERAL ---
# Lista de modelos de Sentence Transformers que quieres probar
MODELS_TO_TEST = [
    'all-MiniLM-L6-v2',          # Rápido y bueno para inglés
    'paraphrase-multilingual-mpnet-base-v2', # Bueno para multilingüe
    'sentence-t5-base',           # Basado en T5, puede ser más pesado
    'stsb-roberta-large',
    'distiluse-base-multilingual-cased-v1',
]

SOURCE_DATA_FILE = "customer-support-tickets-en-derfinitivo.csv" # Asegúrate que este es el nombre correcto
METADATA_FILENAME_GLOBAL = "support_cases_metadata_GLOBAL.json" # Un único archivo para todos los metadatos

# --- 1. CARGAR DATOS Y PREPARAR METADATOS (Solo una vez) ---
metadata_store = None
df = None
texts_to_embed = None

if os.path.exists(METADATA_FILENAME_GLOBAL):
    print(f"Cargando metadatos globales desde {METADATA_FILENAME_GLOBAL}...")
    with open(METADATA_FILENAME_GLOBAL, 'r', encoding='utf-8') as f:
        metadata_store_from_file = json.load(f)
    
    # Necesitamos df para 'texts_to_embed', así que lo cargamos también o reconstruimos 'texts_to_embed'
    # Por simplicidad aquí, vamos a asumir que si el metadata existe, ya tenemos el df procesado
    # para no duplicar la lógica de 'texts_to_embed'.
    # En una implementación más robusta, podrías guardar 'texts_to_embed' junto con los metadatos.
    try:
        print(f"Cargando DataFrame original desde {SOURCE_DATA_FILE} para obtener textos...")
        df = pd.read_csv(SOURCE_DATA_FILE)
        # Asegúrate de que la columna 'problem_text' existe o créala
        if 'problem_text' not in df.columns:
             if 'subject' in df.columns and 'body' in df.columns:
                df['problem_text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
             else:
                raise ValueError("No se pueden encontrar las columnas 'subject'/'body' para crear 'problem_text'")
        texts_to_embed = df['problem_text'].tolist()
        
        # Reconstruimos metadata_store desde df para asegurar consistencia de IDs con texts_to_embed
        # Esto es más seguro si el orden o contenido de METADATA_FILENAME_GLOBAL pudiera diferir.
        print("Re-generando metadata_store desde el DataFrame para consistencia...")
        metadata_store = []
        for i in range(len(df)):
            row = df.iloc[i]
            metadata_store.append({
                'id': i,
                'original_id_from_df': row.get('Ticket ID', i), # Ejemplo si tienes un ID original
                'subject': row.get('subject', 'N/A'),
                'body': row.get('body', 'N/A'),
                'answer': row.get('answer', 'N/A'),
                'tags': row.get('tags', []) # Asegúrate que 'tags' existe y es una lista o procesa aquí
            })
        print(f"Metadatos (re)generados para {len(metadata_store)} casos.")

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de datos fuente: {SOURCE_DATA_FILE}, necesario para los textos.")
        exit()
    except Exception as e:
        print(f"Error al procesar DataFrame para textos: {e}")
        exit()

else:
    print(f"Archivo de metadatos globales {METADATA_FILENAME_GLOBAL} no encontrado. Creándolo...")
    try:
        df = pd.read_csv(SOURCE_DATA_FILE)
        # ---- Preprocesamiento del DataFrame ----
        if 'problem_text' not in df.columns:
             if 'subject' in df.columns and 'body' in df.columns:
                df['problem_text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
             else:
                raise ValueError("No se pueden encontrar las columnas 'subject'/'body' para crear 'problem_text'")

        # Asegúrate que 'tags' es una lista o procesa aquí
        # Ejemplo: if 'tags' in df.columns and isinstance(df['tags'].iloc[0], str):
        #             df['tags'] = df['tags'].fillna('').apply(lambda x: [tag.strip() for tag in x.split(',') if tag.strip()])
        #          elif 'tags' not in df.columns:
        #             df['tags'] = [[] for _ in range(len(df))]


        texts_to_embed = df['problem_text'].tolist()
        print(f"Datos originales cargados y textos para embed preparados: {len(df)} filas.")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de datos fuente: {SOURCE_DATA_FILE}")
        exit()
    except Exception as e:
        print(f"Error al procesar DataFrame: {e}")
        exit()

    metadata_store = []
    for i in range(len(df)):
        row = df.iloc[i]
        metadata_store.append({
            'id': i, # ID interno para FAISS (0 a N-1)
            'original_id_from_df': row.get('Ticket ID', i), # Ejemplo si tienes un ID original en el CSV
            'subject': row.get('subject', 'N/A'),
            'body': row.get('body', 'N/A'),
            'answer': row.get('answer', 'N/A'),
            'tags': row.get('tags', []) # Asegúrate que 'tags' es una lista o se procesa correctamente
        })
    
    try:
        with open(METADATA_FILENAME_GLOBAL, 'w', encoding='utf-8') as f:
            json.dump(metadata_store, f, ensure_ascii=False, indent=4)
        print(f"Metadatos globales guardados en {METADATA_FILENAME_GLOBAL} para {len(metadata_store)} casos.")
    except Exception as e:
        print(f"Error al guardar metadatos globales: {e}")


# --- 2. BUCLE PARA PROBAR CADA MODELO ---
for model_name in MODELS_TO_TEST:
    print(f"\n--- Procesando Modelo: {model_name} ---")

    # Nombres de archivo específicos para el índice FAISS de este modelo
    sanitized_model_name = model_name.replace('/', '_').replace('-', '_') # Crear un nombre de archivo seguro
    index_filename_for_model = f"support_cases_faiss_{sanitized_model_name}.index"

    current_embedding_model = None
    current_index = None

    try:
        print(f"Cargando modelo de embedding: {model_name}...")
        current_embedding_model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"Error al cargar el modelo {model_name}: {e}. Saltando este modelo.")
        continue # Saltar al siguiente modelo

    if os.path.exists(index_filename_for_model):
        print(f"Cargando índice FAISS existente para {model_name} desde {index_filename_for_model}...")
        try:
            current_index = faiss.read_index(index_filename_for_model)
            print(f"Índice cargado. Contiene {current_index.ntotal} vectores.")
            if current_index.d != current_embedding_model.get_sentence_embedding_dimension():
                print(f"Advertencia: La dimensión del índice cargado ({current_index.d}) no coincide con la del modelo ({current_embedding_model.get_sentence_embedding_dimension()}). Esto podría causar errores.")
                print("Se recomienda eliminar el archivo .index y regenerarlo.")
                # Podrías optar por forzar la regeneración: current_index = None 
        except Exception as e:
            print(f"Error al cargar el índice {index_filename_for_model}: {e}. Se intentará regenerar.")
            current_index = None # Forzar regeneración
    
    if current_index is None: # Si no existía o falló la carga
        print(f"Generando nuevo índice FAISS para {model_name}...")
        try:
            embeddings = current_embedding_model.encode(texts_to_embed, show_progress_bar=True)
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)
            
            print(f"Normalizando embeddings para {model_name}...")
            faiss.normalize_L2(embeddings) # Normaliza L2 in-place

            d = embeddings.shape[1]
            current_index = faiss.IndexFlatIP(d)
            current_index.add(embeddings)
            print(f"Índice FAISS construido para {model_name}. Total vectores: {current_index.ntotal}")

            print(f"Guardando índice FAISS para {model_name} en {index_filename_for_model}...")
            faiss.write_index(current_index, index_filename_for_model)
        except Exception as e:
            print(f"Error al generar/guardar el índice para {model_name}: {e}. Saltando este modelo.")
            continue


    # --- FUNCIÓN DE BÚSQUEDA (Adaptada para tomar modelo e índice) ---
    def search_similar_cases_dynamic(query_text, emb_model, faiss_index, meta_store, k=5):
        if not emb_model or not faiss_index or not meta_store:
            print("Error: Modelo, índice o metadatos no disponibles para la búsqueda.")
            return []
        try:
            query_embedding = emb_model.encode([query_text])[0].astype(np.float32)
            query_embedding_norm = np.copy(query_embedding)
            faiss.normalize_L2(query_embedding_norm.reshape(1, -1))
            query_vector = query_embedding_norm.reshape(1, -1)
            
            distances, indices = faiss_index.search(query_vector, k)
            
            results = []
            if indices.size == 0 or indices[0][0] == -1 : return results

            for i in range(min(k, len(indices[0]))):
                result_index = indices[0][i]
                if result_index == -1: continue
                if 0 <= result_index < len(meta_store):
                    similarity_score = distances[0][i]
                    result_metadata = meta_store[result_index]
                    results.append({
                        'id': result_metadata.get('id', result_index),
                        'score': similarity_score,
                        'subject': result_metadata.get('subject', 'N/A'),
                        'body': result_metadata.get('body', 'N/A'),
                        'answer': result_metadata.get('answer', 'N/A'),
                        'tags': result_metadata.get('tags', [])
                    })
                else:
                     print(f"Advertencia: Índice FAISS {result_index} fuera de rango para metadatos (tamaño {len(meta_store)}).")
            return results
        except Exception as e:
            print(f"Error durante la búsqueda con el modelo {emb_model.model_name if hasattr(emb_model, 'model_name') else 'desconocido'}: {e}")
            return []

    # --- PROBAR LA BÚSQUEDA PARA EL MODELO ACTUAL ---
    if current_index and metadata_store and current_embedding_model:
        test_query = "I'm having problems with the encryption of my data" # Puedes usar varias queries de prueba
        print(f"\nResultados para la consulta: '{test_query}' (Usando Modelo: {model_name})")
        search_results = search_similar_cases_dynamic(test_query, current_embedding_model, current_index, metadata_store, k=3) # k=3 para brevedad

        if not search_results:
            print("  No se encontraron resultados similares.")
        else:
            for result in search_results:
                print(f"  --- ID: {result['id']} (Score: {result['score']:.4f}) ---")
                print(f"    Subject: {result['subject']}")
                # print(f"    Tags: {result['tags']}")
                # print(f"    Body: {result['body']}")
                # print(f"    Answer: {result['answer']}")
    else:
        print(f"No se pudo realizar la búsqueda para el modelo {model_name} debido a errores previos.")

print("\n--- Proceso de prueba de modelos completado. ---")

# No hay un guardado global al final, ya que cada índice se guarda
# individualmente y los metadatos se guardaron una vez al principio.