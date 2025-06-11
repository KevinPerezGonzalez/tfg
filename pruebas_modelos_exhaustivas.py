import os
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import json
import faiss
import time # To measure processing time
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURACIÓN GENERAL ---
MODELS_TO_TEST = [
    'all-MiniLM-L6-v2',          # Rápido y bueno para inglés
    'paraphrase-multilingual-mpnet-base-v2', # Bueno para multilingüe
    'sentence-t5-base',            # Basado en T5
    'stsb-roberta-large',
    'distiluse-base-multilingual-cased-v1', 
]

SOURCE_DATA_FILE = "customer-support-tickets-en-derfinitivo.csv"
METADATA_FILENAME_GLOBAL = "support_cases_metadata_GLOBAL.json"
COMPARISON_RESULTS_FILE = "model_comparison_results.csv" # Para guardar los resultados de la comparación

# --- QUERIES DE PRUEBA ---
# (original_expected_id se usa para referencia y posible análisis de auto-recuperación si el ID está en tus datos)
TEST_QUERIES = [
    {'query_id': 157, 'text': "Inconsistencies in investment analytics dashboard data"},
    {'query_id': 448, 'text': "Integration issues between Smart-Gartenbewässerung, Slack and Elasticsearch"},
    {'query_id': 636, 'text': "Users experiencing occasional login difficulties on project platform"},
    {'query_id': 674, 'text': "Urgent data breach affecting medical records"},
    {'query_id': 689, 'text': "Encryption failure due to software problems"},
    {'query_id': 790, 'text': "Digital strategies for brand expansion inquiry"},
    {'query_id': 1195, 'text': "Suspected security breach with unauthorized medical data access"},
    {'query_id': 1372, 'text': "Persistent login problem despite cache clearing"},
    {'query_id': 1479, 'text': "Data analytics disruption after server crash"},
    {'query_id': 1541, 'text': "ClickUp integration causing project management delays"},
    {'query_id': 1754, 'text': "HubSpot CRM integration compatibility issues"},
    {'query_id': 1764, 'text': "Investment optimization tools malfunctioning post-update"},
    {'query_id': 1838, 'text': "Login problems possibly due to outdated credentials"},
    {'query_id': 1864, 'text': "Inquiry about secure medical data storage updates"},
    {'query_id': 1904, 'text': "Request for investment optimization analytics services info"},
    {'query_id': 1907, 'text': "Magento 2.4 integration assistance for digital marketing"},
    {'query_id': 1946, 'text': "Keras integration documentation request for project management SaaS"},
    {'query_id': 1970, 'text': "Security best practices for medical data with RapidMiner and AWS"},
    {'query_id': 1984, 'text': "Investment model producing suboptimal recommendations due to outdated data"},
    {'query_id': 2001, 'text': "Campaign data vanished after Node.js update"},
    {'query_id': 2040, 'text': "Request for advanced data visualization tool upgrades"},
    {'query_id': 2164, 'text': "Connectivity problems with project management SaaS solution"},
    {'query_id': 2187, 'text': "Ad campaign underperformance due to targeting errors"},
    {'query_id': 2204, 'text': "Login difficulties following system update"},
    {'query_id': 2546, 'text': "Billing problem with subscription renewal"},
    {'query_id': 2597, 'text': "Security measures recommendation for hospital patient data"},
    {'query_id': 2715, 'text': "Decline in brand engagement despite digital strategies"},
    {'query_id': 2784, 'text': "Data analytics dashboard unexpected crash"},
    {'query_id': 2915, 'text': "Client engagement rates decrease in marketing agency"},
    {'query_id': 3006, 'text': "Digital brand growth strategies inquiry"},
    {'query_id': 3035, 'text': "Request for SaaS platform integration enhancements"},
    {'query_id': 3157, 'text': "Router connection issues affecting multiple applications"},
    {'query_id': 3337, 'text': "Multiple integrations failing after Docker updates"},
    {'query_id': 3389, 'text': "Campaign metrics not updating due to API issues"},
    {'query_id': 3477, 'text': "Recurring access problems with SaaS application"},
    {'query_id': 3614, 'text': "Securing medical data in hospital IT systems inquiry"},
    {'query_id': 3799, 'text': "Project data vanished after DocuSign integration"},
    {'query_id': 3861, 'text': "Medical records exposed in data breach from insecure plugin"},
    {'query_id': 3865, 'text': "Service outages in digital marketing tools"},
    {'query_id': 3991, 'text': "Unexpected billing error affecting multiple products"},
    {'query_id': 4056, 'text': "Healthcare provider security breaches from outdated software"},
    {'query_id': 4207, 'text': "Payment options inquiry for subscription plans"},
    {'query_id': 4223, 'text': "ClickUp Excel integration compatibility issues"},
    {'query_id': 4351, 'text': "SaaS application unanticipated crashes causing downtime"},
    {'query_id': 4414, 'text': "Investment strategies using data analytics inquiry"},
    {'query_id': 4521, 'text': "Digital campaigns halted due to Firebase integration issue"},
    {'query_id': 4814, 'text': "Headset billing options request"},
    {'query_id': 4919, 'text': "Recurring software disruptions during peak traffic"},
    {'query_id': 4924, 'text': "Software integration failure after updates"},
    {'query_id': 5046, 'text': "Unanticipated service interruptions in data analytics tools"},
    {'query_id': 5282, 'text': "Smartsheet sync issues after software updates"},
    {'query_id': 5361, 'text': "Workstation encryption failure possibly from software incompatibility"},
    {'query_id': 5810, 'text': "Malware detected on Kodak scanner due to outdated macOS"},
    {'query_id': 5985, 'text': "Data analytics tools failing to generate reports"},
    {'query_id': 6170, 'text': "Urgent healthcare system security breach exposing medical data"},
    {'query_id': 6275, 'text': "Git sync failure in Azure from incompatible Node.js version"},
    {'query_id': 6409, 'text': "Frequent crashes of investment optimization tool during analysis"},
    {'query_id': 6435, 'text': "Unable to access recent project updates due to server glitch"},
    {'query_id': 6447, 'text': "Unexpected drop in digital marketing campaign engagement"},
    {'query_id': 6452, 'text': "Project data vanished possibly from sync issues"},
    {'query_id': 6749, 'text': "Application crashing during peak usage hours"},
    {'query_id': 6982, 'text': "Frequent system crashes possibly from resource overload"},
    {'query_id': 7017, 'text': "Project management software system requirements inquiry"},
    {'query_id': 7227, 'text': "Medical data access denied possibly from malware"},
    {'query_id': 7352, 'text': "Request for UI update to improve navigation experience"},
    {'query_id': 7419, 'text': "Digital campaign underperformance from inconsistent tracking"},
    {'query_id': 7584, 'text': "Medical data security services details request"},
    {'query_id': 7745, 'text': "Elasticsearch periodic failures after update on Mac Mini"},
    {'query_id': 7766, 'text': "Medical data encryption failure on iMac"},
    {'query_id': 7942, 'text': "Decline in digital campaign engagement despite adjustments"},
    {'query_id': 7993, 'text': "Connectivity issues with SaaS project management tool"},
    {'query_id': 7994, 'text': "Unauthorized access attempts to health records networks"},
    {'query_id': 8002, 'text': "Unauthorized access incident to medical data"},
    {'query_id': 8137, 'text': "Medical data security measures inquiry for Oracle Database"},
    {'query_id': 8231, 'text': "Software crashing and generating errors after updates"},
    {'query_id': 8355, 'text': "Website analytics tool malfunction due to integration issue"},
    {'query_id': 8399, 'text': "Digital brand expansion techniques inquiry"},
    {'query_id': 8683, 'text': "Securing medical data in PostgreSQL for hospital environment"},
    {'query_id': 8714, 'text': "Multiple tools malfunctioning after recent update"},
    {'query_id': 8735, 'text': "Digital strategies for tech product brand growth"},
    {'query_id': 8777, 'text': "Delayed investment analysis results from data integration issues"},
    {'query_id': 8854, 'text': "Investment predictions failed due to data processing errors"},
    {'query_id': 8884, 'text': "Request to enhance medical data security protocols"},
    {'query_id': 9004, 'text': "Update billing details for marketing firm"},
    {'query_id': 9134, 'text': "Campaign analytics dashboard failure from outdated Ruby version"},
    {'query_id': 9252, 'text': "Elasticsearch 7.13 integration details for SaaS project management"},
    {'query_id': 9316, 'text': "Intermittent connectivity issues across multiple products"},
    {'query_id': 9420, 'text': "Marketing agency website downtime from possible server overload"},
    {'query_id': 9569, 'text': "Security measures for confidential medical information"},
    {'query_id': 9703, 'text': "Project management workflow issues needing integration updates"},
    {'query_id': 9789, 'text': "Request to revise maintenance schedule to reduce peak hour downtime"},
    {'query_id': 9929, 'text': "Securing medical data through Smart-Steckdose QuickBooks integration"},
    {'query_id': 10090, 'text': "Web traffic decline possibly from algorithm change"},
    {'query_id': 10092, 'text': "Underperforming marketing campaigns despite adjustments"},
    {'query_id': 10149, 'text': "Online engagement decrease despite marketing efforts"},
    {'query_id': 10181, 'text': "Investment optimization using data analytics tools inquiry"},
    {'query_id': 10479, 'text': "Data import failure due to file format issues"},
    {'query_id': 10588, 'text': "Security protocols for OLED monitors in medical facilities"},
    {'query_id': 10745, 'text': "Occasional login failures on project platform"},
    {'query_id': 10862, 'text': "System data breach with subsequent security updates"}
]
TOP_K_RESULTS_TO_ANALYZE = 3 # Cuántos resultados por consulta guardar para análisis

# --- 1. CARGAR DATOS Y PREPARAR METADATOS (Solo una vez) ---
metadata_store = None
df = None
texts_to_embed = None

# Esta función de ayuda procesa el DataFrame y crea metadata_store y texts_to_embed
def load_and_prepare_data(source_file):
    print(f"Cargando y procesando DataFrame desde {source_file}...")
    try:
        current_df = pd.read_csv(source_file)
        # Preprocesamiento del DataFrame (asegurar que 'problem_text' y 'tags' existen)
        if 'problem_text' not in current_df.columns:
            if 'subject' in current_df.columns and 'body' in current_df.columns:
                current_df['problem_text'] = current_df['subject'].fillna('') + ' ' + current_df['body'].fillna('')
            else:
                raise ValueError("Columnas 'subject'/'body' no encontradas para crear 'problem_text'")

        # Manejo de 'tags': Asegurar que es una lista. Si es string separado por comas:
        if 'tags' in current_df.columns and current_df['tags'].notna().any() and isinstance(current_df['tags'].dropna().iloc[0], str):
            print("Convirtiendo columna 'tags' de string a lista...")
            current_df['tags'] = current_df['tags'].fillna('').apply(
                lambda x: [tag.strip() for tag in x.split(',') if tag.strip()] if isinstance(x, str) else []
            )
        elif 'tags' not in current_df.columns:
            current_df['tags'] = [[] for _ in range(len(current_df))]
        
        current_texts_to_embed = current_df['problem_text'].tolist()
        
        current_metadata_store = []
        for i in range(len(current_df)):
            row = current_df.iloc[i]
            current_metadata_store.append({
                'doc_id_internal': i, # ID interno para FAISS (0 a N-1)
                'original_ticket_id': row.get('Ticket ID', row.get('id', i)), # Prioriza 'Ticket ID', luego 'id', luego índice
                'subject': row.get('subject', 'N/A'),
                'body': row.get('body', 'N/A'),
                'answer': row.get('answer', 'N/A'),
                'tags': row.get('tags', [])
            })
        print(f"DataFrame cargado y procesado. {len(current_df)} filas. {len(current_metadata_store)} metadatos.")
        return current_df, current_metadata_store, current_texts_to_embed
    except FileNotFoundError:
        print(f"Error CRÍTICO: No se encontró el archivo de datos fuente: {source_file}")
        exit()
    except Exception as e:
        print(f"Error CRÍTICO al procesar DataFrame: {e}")
        exit()

if os.path.exists(METADATA_FILENAME_GLOBAL):
    print(f"Cargando metadatos globales desde {METADATA_FILENAME_GLOBAL}...")
    # Aunque carguemos metadatos, necesitamos df y texts_to_embed para los embeddings.
    # Así que siempre recargamos/reprocesamos df para asegurar consistencia.
    df, metadata_store, texts_to_embed = load_and_prepare_data(SOURCE_DATA_FILE)
    # Podríamos comparar metadata_store cargado con el regenerado si quisiéramos,
    # pero para este script, regenerar desde df asegura que todo está alineado.
    # Si METADATA_FILENAME_GLOBAL es la fuente de verdad absoluta, la lógica sería diferente.
    # Aquí, el CSV es la fuente de verdad, METADATA_FILENAME_GLOBAL es solo un caché.
    # Vamos a sobreescribirlo para asegurar que está actualizado con el CSV y el procesamiento actual.
    print(f"Metadatos globales (re)cargados/generados. Guardando en {METADATA_FILENAME_GLOBAL} para asegurar frescura...")
    try:
        with open(METADATA_FILENAME_GLOBAL, 'w', encoding='utf-8') as f:
            json.dump(metadata_store, f, ensure_ascii=False, indent=4)
        print(f"Metadatos globales actualizados y guardados en {METADATA_FILENAME_GLOBAL}.")
    except Exception as e:
        print(f"Error al guardar metadatos globales actualizados: {e}")
else:
    print(f"Archivo de metadatos globales {METADATA_FILENAME_GLOBAL} no encontrado. Creándolo...")
    df, metadata_store, texts_to_embed = load_and_prepare_data(SOURCE_DATA_FILE)
    try:
        with open(METADATA_FILENAME_GLOBAL, 'w', encoding='utf-8') as f:
            json.dump(metadata_store, f, ensure_ascii=False, indent=4)
        print(f"Metadatos globales guardados en {METADATA_FILENAME_GLOBAL}.")
    except Exception as e:
        print(f"Error al guardar metadatos globales: {e}")

# --- Lista para almacenar todos los resultados de la comparación ---
all_comparison_results = []

# --- 2. BUCLE PARA PROBAR CADA MODELO ---
for model_name in MODELS_TO_TEST:
    start_time_model = time.time()
    print(f"\n{'='*15} Procesando Modelo: {model_name} {'='*15}")

    sanitized_model_name = model_name.replace('/', '_').replace('-', '_')
    index_filename_for_model = f"support_cases_faiss_{sanitized_model_name}.index"

    current_embedding_model = None
    current_index = None

    try:
        print(f"Cargando modelo de embedding: {model_name}...")
        current_embedding_model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"Error al cargar el modelo {model_name}: {e}. Saltando este modelo.")
        continue

    if os.path.exists(index_filename_for_model):
        print(f"Cargando índice FAISS existente para {model_name} desde {index_filename_for_model}...")
        try:
            current_index = faiss.read_index(index_filename_for_model)
            # Verificación de consistencia
            if current_index.ntotal != len(texts_to_embed):
                 print(f"Advertencia: El número de vectores en el índice ({current_index.ntotal}) no coincide con los textos a embeber ({len(texts_to_embed)}). Se regenerará el índice.")
                 current_index = None # Forzar regeneración
            elif current_index.d != current_embedding_model.get_sentence_embedding_dimension():
                print(f"Advertencia: Dimensión del índice cargado ({current_index.d}) no coincide con la del modelo ({current_embedding_model.get_sentence_embedding_dimension()}). Se regenerará el índice.")
                current_index = None # Forzar regeneración
            else:
                print(f"Índice cargado. Contiene {current_index.ntotal} vectores.")
        except Exception as e:
            print(f"Error al cargar el índice {index_filename_for_model}: {e}. Se intentará regenerar.")
            current_index = None
    
    if current_index is None:
        print(f"Generando nuevo índice FAISS para {model_name}...")
        try:
            print(f"Generando embeddings con {model_name} (esto puede tardar)...")
            start_time_embedding = time.time()
            embeddings = current_embedding_model.encode(texts_to_embed, show_progress_bar=True, batch_size=32) # Ajustar batch_size según memoria
            end_time_embedding = time.time()
            print(f"Embeddings generados en {end_time_embedding - start_time_embedding:.2f} segundos.")

            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)
            
            print(f"Normalizando embeddings para {model_name}...")
            faiss.normalize_L2(embeddings)

            d = embeddings.shape[1]
            current_index = faiss.IndexFlatIP(d)
            current_index.add(embeddings)
            print(f"Índice FAISS construido para {model_name}. Total vectores: {current_index.ntotal}")

            print(f"Guardando índice FAISS para {model_name} en {index_filename_for_model}...")
            faiss.write_index(current_index, index_filename_for_model)
        except Exception as e:
            print(f"Error al generar/guardar el índice para {model_name}: {e}. Saltando este modelo.")
            continue

    # --- FUNCIÓN DE BÚSQUEDA (Adaptada) ---
    def search_similar_cases_dynamic(query_text, emb_model, faiss_index, meta_store, k_results=10):
        # ... (la función search_similar_cases_dynamic que ya tenías, pero usa k_results) ...
        if not emb_model or not faiss_index or not meta_store: return []
        try:
            query_embedding = emb_model.encode([query_text])[0].astype(np.float32)
            query_embedding_norm = np.copy(query_embedding)
            faiss.normalize_L2(query_embedding_norm.reshape(1, -1))
            query_vector = query_embedding_norm.reshape(1, -1)
            
            distances, indices = faiss_index.search(query_vector, k_results)
            
            results = []
            if indices.size == 0 or (len(indices[0]) > 0 and indices[0][0] == -1) : return results

            for i in range(min(k_results, len(indices[0]))):
                result_index = indices[0][i]
                if result_index == -1: continue
                if 0 <= result_index < len(meta_store):
                    similarity_score = distances[0][i]
                    retrieved_metadata = meta_store[result_index]
                    results.append({
                        'retrieved_doc_id_internal': retrieved_metadata.get('doc_id_internal', result_index),
                        'retrieved_original_ticket_id': retrieved_metadata.get('original_ticket_id', 'N/A'),
                        'score': similarity_score,
                        'subject': retrieved_metadata.get('subject', 'N/A'),
                        # 'body': retrieved_metadata.get('body', 'N/A'), # Descomentar si se necesita
                        # 'answer': retrieved_metadata.get('answer', 'N/A'),
                        'tags': retrieved_metadata.get('tags', [])
                    })
                else:
                     print(f"Advertencia: Índice FAISS {result_index} fuera de rango para metadatos (tamaño {len(meta_store)}).")
            return results
        except Exception as e:
            model_identifier = model_name # Usar el model_name del bucle exterior
            print(f"Error durante la búsqueda con el modelo {model_identifier}: {e}")
            return []

    # --- PROBAR LA BÚSQUEDA PARA EL MODELO ACTUAL CON TODAS LAS QUERIES DE PRUEBA ---
    if current_index and metadata_store and current_embedding_model:
        print(f"\nEjecutando queries de prueba para el modelo: {model_name}...")
        for test_query_item in TEST_QUERIES:
            query_id = test_query_item['query_id']
            query_text = test_query_item['text']
            
            # print(f"  Buscando para Query ID {query_id}: '{query_text}'")
            search_results = search_similar_cases_dynamic(query_text, current_embedding_model, current_index, metadata_store, k_results=TOP_K_RESULTS_TO_ANALYZE)

            if not search_results:
                # print(f"    No se encontraron resultados.")
                all_comparison_results.append({
                    'model_name': model_name,
                    'query_id': query_id,
                    'query_text': query_text,
                    'rank': 0, # Indica que no hubo resultados válidos
                    'retrieved_doc_id_internal': None,
                    'retrieved_original_ticket_id': None,
                    'retrieved_subject': "NO RESULTS",
                    'score': np.nan, # Usar NaN para scores no existentes
                    'is_self_retrieval': False # (asumiendo que el original_ticket_id se puede mapear)
                })
            else:
                for rank, result in enumerate(search_results):
                    # Chequeo de auto-recuperación (si el 'original_ticket_id' del resultado coincide con 'query_id')
                    is_self = result.get('retrieved_original_ticket_id') == query_id
                    
                    all_comparison_results.append({
                        'model_name': model_name,
                        'query_id': query_id,
                        'query_text': query_text,
                        'rank': rank + 1,
                        'retrieved_doc_id_internal': result['retrieved_doc_id_internal'],
                        'retrieved_original_ticket_id': result['retrieved_original_ticket_id'],
                        'retrieved_subject': result['subject'],
                        'score': result['score'],
                        'is_self_retrieval': is_self
                    })
    else:
        print(f"No se pudo realizar la búsqueda para el modelo {model_name} debido a errores previos.")
    
    end_time_model = time.time()
    print(f"Tiempo total para el modelo {model_name}: {end_time_model - start_time_model:.2f} segundos.")


print("\n{'='*15} Proceso de prueba de modelos completado. {'='*15}")

# --- 3. ANÁLISIS Y PRESENTACIÓN DE RESULTADOS DE COMPARACIÓN ---
if all_comparison_results:
    comparison_df = pd.DataFrame(all_comparison_results)
    print(f"\nGuardando resultados detallados de la comparación en: {COMPARISON_RESULTS_FILE}")
    try:
        comparison_df.to_csv(COMPARISON_RESULTS_FILE, index=False, encoding='utf-8-sig')
        print("Resultados guardados.")
    except Exception as e:
        print(f"Error al guardar {COMPARISON_RESULTS_FILE}: {e}")

    print("\n--- Resumen de Rendimiento por Modelo ---")
    
    # Calcular score promedio del primer resultado (rank 1) por modelo
    avg_top1_score = comparison_df[comparison_df['rank'] == 1].groupby('model_name')['score'].mean().sort_values(ascending=False)
    print("\nScore Promedio del Primer Resultado (Top 1) por Modelo (mayor es mejor para IP/Coseno):")
    print(avg_top1_score)

    # Calcular tasa de auto-recuperación en el Top 1 (si 'original_ticket_id' es fiable)
    # Esto solo es útil si los query_id de TEST_QUERIES realmente corresponden a 'original_ticket_id' en tus datos
    self_retrieval_top1_rate = comparison_df[
        (comparison_df['rank'] == 1) & (comparison_df['is_self_retrieval'] == True)
    ].groupby('model_name').size() / len(TEST_QUERIES) # Tasa sobre el total de queries de prueba
    self_retrieval_top1_rate = self_retrieval_top1_rate.sort_values(ascending=False)
    print("\nTasa de Auto-Recuperación en el Top 1 por Modelo (si aplica):")
    print(self_retrieval_top1_rate)
    
    # Imprimir los mejores resultados para cada query para cada modelo (para inspección visual)
    print("\n--- Mejores Resultados (Top 1) por Query y Modelo para Inspección ---")
    top1_results = comparison_df[comparison_df['rank'] == 1]
    for query_item in TEST_QUERIES:
        qid = query_item['query_id']
        qtext = query_item['text']
        print(f"\n  Query ID {qid}: '{qtext}'")
        results_for_query = top1_results[top1_results['query_id'] == qid].sort_values(by='score', ascending=False)
        for _, row in results_for_query.iterrows():
            print(f"    Modelo: {row['model_name']:<40} Score: {row['score']:.4f} -> ID recuperado: {row['retrieved_original_ticket_id']} (S: {row['is_self_retrieval']}) Sub: {row['retrieved_subject'][:60]}...")

    # --- NUEVO: GRÁFICOS COMPARATIVOS ---
    print("\nGenerando gráficos comparativos...")
    
    # Configuración estética
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    # 1. Gráfico de scores de los 10 mejores resultados por modelo
    TOP_N_FOR_PLOTS = 10  # Número de resultados a considerar para los gráficos
    
    # Filtrar los TOP_N mejores resultados para cada modelo
    top_n_results = comparison_df[comparison_df['rank'] <= TOP_N_FOR_PLOTS]
    
    # Crear directorio para guardar gráficos si no existe
    os.makedirs("comparison_plots", exist_ok=True)
    
    # Gráfico 1: Distribución de scores por modelo (top 10)
    plt.figure(figsize=(14, 8))
    ax = sns.boxplot(x='model_name', y='score', data=top_n_results, 
                    hue='model_name', palette="viridis", legend=False)
    ax.set_title(f'Distribución de Scores para los {TOP_N_FOR_PLOTS} Mejores Resultados por Modelo', pad=20)
    ax.set_xlabel('Modelo')
    ax.set_ylabel('Score de Similitud (coseno)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plot1_path = os.path.join("comparison_plots", "model_scores_distribution.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfico 2: Media de scores de los 10 mejores por modelo
    plt.figure(figsize=(14, 6))
    mean_scores = top_n_results.groupby('model_name')['score'].mean().sort_values(ascending=False)
    ax = sns.barplot(x=mean_scores.index, y=mean_scores.values, 
                    hue=mean_scores.index, palette="rocket", legend=False)
    ax.set_title(f'Media de Scores para los {TOP_N_FOR_PLOTS} Mejores Resultados por Modelo', pad=20)
    ax.set_xlabel('Modelo')
    ax.set_ylabel('Score Promedio')
    plt.xticks(rotation=45, ha='right')
    
    # Añadir los valores numéricos en las barras
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points')
    plt.tight_layout()
    plot2_path = os.path.join("comparison_plots", "model_mean_scores_comparison.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfico 3: Scores de los 10 mejores resultados por modelo (puntos)
    plt.figure(figsize=(14, 8))
    ax = sns.stripplot(x='model_name', y='score', data=top_n_results,
                    hue='model_name', palette="viridis", 
                    jitter=0.25, size=6, alpha=0.7, legend=False)
    ax.set_title(f'Scores Individuales para los {TOP_N_FOR_PLOTS} Mejores Resultados por Modelo', pad=20)
    ax.set_xlabel('Modelo')
    ax.set_ylabel('Score de Similitud (coseno)')
    plt.xticks(rotation=45, ha='right')
    
    # Añadir línea horizontal para la media
    for i, model in enumerate(mean_scores.index):
        ax.axhline(y=mean_scores[model], xmin=i/len(mean_scores), xmax=(i+1)/len(mean_scores), 
                  color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plot3_path = os.path.join("comparison_plots", "model_scores_scatter.png")
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico 3 guardado en: {plot3_path}")
    plt.close()
    
    print("\nGráficos generados exitosamente en la carpeta 'comparison_plots'")

else:
    print("No se generaron resultados de comparación.")