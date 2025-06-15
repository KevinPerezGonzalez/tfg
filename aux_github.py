import json
import os
import re
import time
from datetime import datetime, timedelta
import faiss
import numpy as np
import pandas as pd
from github import Github, RateLimitExceededException
from apscheduler.schedulers.background import BackgroundScheduler
from sentence_transformers import SentenceTransformer

# --- CONFIGURACIÓN ---
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
REPO_NAME = "microsoft/PowerToys"

# Lista de modelos para los que se generarán/actualizarán los índices
MODELS_TO_PROCESS = [
    'sentence-t5-base',
]

# Nombres de archivos de salida
SOURCE_DATA_FILE = "powertoys_structured_tickets.csv" # El CSV que se actualiza
TIMESTAMP_FILE = "powertoys_structured_last_update.txt"
METADATA_FILENAME_GLOBAL = "powertoys_metadata_GLOBAL.json" # El JSON para la API


# --- FUNCIONES DE AYUDA ---

def parse_powertoys_issue_body(body_text):
    """
    Parsea el cuerpo de una issue de PowerToys para extraer campos estructurados.
    Devuelve un diccionario con los datos extraídos.
    """
    if not body_text:
        return {}

    # Diccionario para mapear las cabeceras (incluyendo emojis) a claves sencillas
    headers_map = {
        "Microsoft PowerToys version": "version",
        "Installation method": "install_method",
        "Area(s) with issue?": "area",
        "Steps to reproduce": "steps_to_reproduce",
        "✔️ Expected Behavior": "expected_behavior",
        "❌ Actual Behavior": "actual_behavior",
        "Additional Information": "additional_info",
        "Other Software": "other_software"
    }
    
    # Preparamos una expresión regular que buscará todas nuestras cabeceras
    # Esto divide el texto en segmentos basados en las cabeceras que definimos
    headers_pattern = "|".join(f"### {re.escape(h)}" for h in headers_map.keys())
    
    # Usamos re.split para dividir el cuerpo por las cabeceras. El primer elemento suele ser vacío.
    segments = re.split(f"({headers_pattern})", body_text)
    
    data = {}
    # Empezamos desde el índice 1, ya que el primer elemento de la división es el texto antes de la primera cabecera
    for i in range(1, len(segments), 2):
        header_full = segments[i].replace("### ", "").strip()
        content = segments[i+1].strip()
        
        # Si el contenido no es "No response" o similar, lo guardamos
        if content and content.lower() not in ["no response", "n/a"]:
            # Usamos nuestro mapa para obtener una clave limpia
            field_key = headers_map.get(header_full)
            if field_key:
                data[field_key] = content

    return data

def parse_bot_comment(comment_body):
    """
    Parsea el comentario del bot para extraer los números de las issues enlazadas y sus scores.
    Devuelve una lista ordenada de tuplas: (issue_number, score)
    """
    pattern = re.compile(r'#(\d+)\s*,\s*similarity score:\s*(\d+)%')
    matches = pattern.findall(comment_body)
    parsed_issues = []
    for issue_num_str, score_str in matches:
        parsed_issues.append((int(issue_num_str), int(score_str)))
    parsed_issues.sort(key=lambda x: x[1], reverse=True)
    return parsed_issues

def get_solution_from_issue(issue):
    """
    Aplica la heurística para encontrar la respuesta de un colaborador en UNA issue.
    Devuelve el cuerpo del comentario de la solución, o None si no se encuentra.
    """
    collaborator_comments = []
    # Usar reversed() por si la solución suele estar al final
    for comment in reversed(list(issue.get_comments())):
        if comment.author_association in ['COLLABORATOR', 'MEMBER', 'OWNER']:
            collaborator_comments.append(comment)
    
    if collaborator_comments:
        return max(collaborator_comments, key=lambda c: c.created_at).body
    return None

def find_final_solution(issue, repo, visited=None, max_hops=3):
    """
    Busca una solución de forma recursiva con la lógica de prioridad correcta y de forma eficiente.
    """
    if visited is None:
        visited = set()

    if issue.number in visited or len(visited) >= max_hops:
        # Imprime solo si se alcanza el límite, no en visitas normales
        #if len(visited) >= max_hops:
            #print(f"  -> Límite de {max_hops} saltos alcanzado al procesar la issue #{issue.number} o se ha encontrado un bucle. Deteniendo esta cadena.")
        return None

    visited.add(issue.number)

    try:
        # Mejora de eficiencia: Obtener todos los comentarios UNA SOLA VEZ.
        all_comments = list(issue.get_comments())
    except Exception as e:
        print(f"    Advertencia: No se pudieron obtener comentarios para la issue #{issue.number}: {e}")
        return None

    # --- RECOLECCIÓN DE PISTAS ---
    canonical_issue_num = None
    bot_linked_issue_num = None
    latest_human_answer = None

    # Iterar para encontrar las pistas de 'dup' y del 'bot'
    for comment in all_comments:
        # Pista de duplicado (si aún no la hemos encontrado)
        if not canonical_issue_num:
            dup_match = re.search(r'(?i)\s*\/?\s*dup(?:licate)?(?: of)?\s*#(\d+)', comment.body)
            if dup_match:
                canonical_issue_num = int(dup_match.group(1))
        
        # Pista del bot (si aún no la hemos encontrado)
        if not bot_linked_issue_num and comment.user.login == 'similar-issues-ai[bot]':
            linked_issues = parse_bot_comment(comment.body)
            if linked_issues:
                bot_linked_issue_num, _ = linked_issues[0]

    # Iterar en orden inverso (del más nuevo al más viejo) para encontrar la respuesta humana más reciente
    # que NO sea un simple comentario de duplicado.
    for comment in reversed(all_comments):
        if comment.author_association in ['COLLABORATOR', 'MEMBER', 'OWNER']:
            # Comprobar que este comentario no sea el que marca el duplicado
            if not re.search(r'(?i)\s*\/?\s*dup(?:licate)?(?: of)?\s*#(\d+)', comment.body):
                latest_human_answer = comment.body
                break # Encontramos la respuesta humana más reciente y válida

    # --- ÁRBOL DE DECISIÓN BASADO EN PRIORIDADES ---

    # Prioridad 1: Seguir la pista del duplicado
    if canonical_issue_num:
        #print(f"  -> P1: Issue #{issue.number} es dup de #{canonical_issue_num}. Siguiendo pista...")
        try:
            next_issue = repo.get_issue(number=canonical_issue_num)
            return find_final_solution(next_issue, repo, visited, max_hops)
        except Exception as e:
            print(f"    No se pudo obtener la issue canónica #{canonical_issue_num}: {e}")
            return None

    # Prioridad 2: Seguir la pista del bot de IA
    if bot_linked_issue_num:
        #print(f"  -> P2: Issue #{issue.number} es similar a #{bot_linked_issue_num}. Siguiendo pista...")
        try:
            next_issue = repo.get_issue(number=bot_linked_issue_num)
            return find_final_solution(next_issue, repo, visited, max_hops)
        except Exception as e:
            print(f"    No se pudo obtener la issue similar #{bot_linked_issue_num}: {e}")
            return None
    
    # Prioridad 3: Usar la respuesta humana directa como último recurso
    if latest_human_answer:
        #print(f"  -> P3: Respuesta humana directa encontrada en la issue #{issue.number}.")
        return (latest_human_answer, issue.number)
            
    # Si ninguna estrategia funcionó
    return None

# --- FUNCIÓN PRINCIPAL DE ACTUALIZACIÓN ---

def update_csv_from_github():
    """
    Obtiene datos de GitHub y actualiza el archivo CSV.
    Devuelve True si se añadieron nuevos datos, False en caso contrario.
    """
    print(f"[{datetime.now()}] Iniciando actualización desde '{REPO_NAME}'...")

    # 1. Autenticación y acceso al repositorio (sin cambios)
    if not GITHUB_TOKEN:
        print("Error: El token de GitHub no está configurado. Saliendo.")
        return
    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(REPO_NAME)

    # 2. Determinar desde cuándo buscar (sin cambios)
    since_timestamp = None
    if os.path.exists(TIMESTAMP_FILE):
        with open(TIMESTAMP_FILE, 'r') as f:
            try:
                since_timestamp = datetime.fromisoformat(f.read().strip()) + timedelta(seconds=1)
            except (ValueError, IndexError):
                print(f"Advertencia: Formato de fecha inválido en {TIMESTAMP_FILE}. Se obtendrá el historial completo.")
    
    # 3. Obtener las issues cerradas y relevantes
    try:
        # --- INICIO DE LA MODIFICACIÓN ---
        # Construimos un diccionario con los parámetros de la llamada a la API
        api_params = {
            'state': 'closed',
            'labels': ['Issue-Bug']
        }
        
        # Solo añadimos el parámetro 'since' si tiene un valor (es decir, no es la primera ejecución)
        if since_timestamp:
            print(f"Buscando issues cerradas en '{REPO_NAME}' desde: {since_timestamp.isoformat()}")
            api_params['since'] = since_timestamp
        else:
            print(f"Primera ejecución. Obteniendo el historial completo de issues de '{REPO_NAME}'.")
            print("ADVERTENCIA: Esto puede tardar mucho tiempo y consumir muchos créditos de la API.")
        
        # Usamos ** para desempaquetar el diccionario como argumentos de la función.
        # Si 'since' no se añadió al diccionario, no se pasará a la función.
        closed_issues = repo.get_issues(**api_params)
        
    except RateLimitExceededException:
        print("Límite de la API de GitHub alcanzado. Inténtalo de nuevo más tarde (en una hora). Saliendo.")
        return
    except Exception as e:
        # Mantenemos el diagnóstico detallado por si acaso
        print(f"Error DETALLADO al obtener issues de la API de GitHub:")
        print(f"  - TIPO DE ERROR: {type(e)}")
        print(f"  - MENSAJE: {e}")
        print(f"  - REPRESENTACIÓN: {repr(e)}")
        return

    # 4. Procesar cada issue.
    new_data = []
    issues_processed_count = 0
    print("Procesando issues (puede tardar si hay muchas)...")
    
    for i, issue in enumerate(closed_issues):

        if (i + 1) % 50 == 0:
            print(f"  ... procesadas {i + 1} issues ...")
        
        try:

            result_tuple = find_final_solution(issue, repo, visited=set())
            
            if result_tuple:

                #print(f"  -> ÉXITO: Solución encontrada para la issue #{issue.number}. Guardando datos.")

                # 1. Parsear el cuerpo de la issue para extraer los campos estructurados
                parsed_body = parse_powertoys_issue_body(issue.body)

                # 2. Construir un 'problem_text' de alta calidad para el embedding
                title = issue.title
                area = parsed_body.get('area', '')
                actual_behavior = parsed_body.get('actual_behavior', '')
                steps = parsed_body.get('steps_to_reproduce', '')
                
                # Combinamos los campos más relevantes para describir el problema
                high_quality_problem_text = f"Área: {area}. Problema: {title}. Comportamiento: {actual_behavior}. Pasos para reproducir: {steps}"

                # 3. Crear el registro para el CSV con los nuevos campos
                new_entry = {
                    'Ticket ID': issue.number,
                    'subject': title,
                    'problem_text': high_quality_problem_text, # Usamos el texto de alta calidad
                    'answer': result_tuple[0],
                    'answer_source_id': result_tuple[1],
                    'is_direct_answer': issue.number == result_tuple[1],
                    'tags': ",".join([label.name for label in issue.labels]),
                    'powertoys_version': parsed_body.get('version'),
                    'install_method': parsed_body.get('install_method'),
                    'area': area,
                    'steps_to_reproduce': steps,
                    'expected_behavior': parsed_body.get('expected_behavior'),
                    'actual_behavior': actual_behavior,
                    'additional_info': parsed_body.get('additional_info')
                }
                new_data.append(new_entry)
                
                issues_processed_count += 1

        except RateLimitExceededException:
            print(f"Límite de la API alcanzado procesando issue #{issue.number}. Deteniendo por ahora.")
            break 
        except Exception as e:
            print(f"No se pudo procesar la issue #{issue.number}: {e}")

    # 5. y 6. Actualizar CSV y timestamp
    print(f"Procesadas {issues_processed_count} nuevas issues con solución encontrada.")

    if new_data:
        new_df = pd.DataFrame(new_data)
        # Guardamos el CSV igualmente para tener un registro completo y persistente
        if os.path.exists(SOURCE_DATA_FILE):
            existing_df = pd.read_csv(SOURCE_DATA_FILE)
            combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['Ticket ID'], keep='last')
            combined_df.to_csv(SOURCE_DATA_FILE, index=False, encoding='utf-8-sig')
        else:
            new_df.to_csv(SOURCE_DATA_FILE, index=False, encoding='utf-8-sig')

        with open(TIMESTAMP_FILE, 'w') as f:
                f.write(datetime.now().isoformat())
        print(f"Nueva fecha de actualización guardada en {TIMESTAMP_FILE}.")
            
        return new_df # <-- Devolvemos solo el DataFrame con los nuevos datos
    else:
        print("No se encontraron nuevas issues procesables. El archivo CSV está actualizado.")

        with open(TIMESTAMP_FILE, 'w') as f:
            f.write(datetime.now().isoformat())
        print(f"Nueva fecha de actualización guardada en {TIMESTAMP_FILE}.")

        return None # <-- Devolvemos None si no hay nada nuevo

def update_artifacts_incrementally(new_data_df, models_list, metadata_json_path):
    """
    Actualiza los metadatos y los índices FAISS de forma incremental.
    Si no existen, los crea desde cero.
    """
    print(f"\n[{datetime.now()}] Iniciando Fase 2: Generación de artefactos (índices y metadatos)...")
    if new_data_df is None or new_data_df.empty:
        print("No hay nuevos datos para añadir a los índices. Proceso finalizado.")
        return

    # Preparar los nuevos textos y metadatos
    new_texts_to_embed = new_data_df['problem_text'].fillna('').tolist()
    new_metadata = new_data_df.to_dict('records')

    for model_name in models_list:
        print(f"\n--- Procesando artefactos para el modelo: {model_name} ---")
        sanitized_model_name = model_name.replace('/', '_').replace('-', '_')
        index_filename = f"faiss_index_{sanitized_model_name}.index"

        try:
            print("Cargando modelo de embedding...")
            model = SentenceTransformer(model_name)

            # Generar embeddings SOLO para los datos nuevos
            print(f"Generando {len(new_texts_to_embed)} nuevos embeddings...")
            new_embeddings = model.encode(new_texts_to_embed, show_progress_bar=True, batch_size=32)
            if new_embeddings.dtype != np.float32:
                new_embeddings = new_embeddings.astype(np.float32)
            faiss.normalize_L2(new_embeddings)

            # Lógica para crear o actualizar el índice y los metadatos
            if os.path.exists(index_filename) and os.path.exists(metadata_json_path):
                # --- LÓGICA DE ACTUALIZACIÓN ---
                print(f"Cargando índice existente desde {index_filename}...")
                index = faiss.read_index(index_filename)
                
                print(f"Cargando metadatos existentes desde {metadata_json_path}...")
                with open(metadata_json_path, 'r', encoding='utf-8') as f:
                    metadata_store = json.load(f)

                # Añadir los nuevos datos
                index.add(new_embeddings)
                metadata_store.extend(new_metadata)

                print(f"Índice actualizado. Total de vectores ahora: {index.ntotal}")
                print(f"Metadatos actualizados. Total de entradas ahora: {len(metadata_store)}")
            else:
                # --- LÓGICA DE CREACIÓN (Primera ejecución) ---
                print("No se encontraron artefactos existentes. Creándolos desde cero...")
                d = new_embeddings.shape[1]
                index = faiss.IndexFlatIP(d)
                index.add(new_embeddings)
                metadata_store = new_metadata
                print(f"Artefactos creados. Total vectores: {index.ntotal}, Total metadatos: {len(metadata_store)}")

            # Guardar los artefactos actualizados/creados
            faiss.write_index(index, index_filename)
            with open(metadata_json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_store, f, ensure_ascii=False, indent=4)
            print(f"Índice y metadatos para '{model_name}' guardados con éxito.")

        except Exception as e:
            print(f"Error procesando el modelo '{model_name}': {e}")
            continue

# --- FUNCIÓN ORQUESTADORA PRINCIPAL ---
def run_complete_pipeline():
    """
    Orquesta todo el proceso: actualizar CSV y luego regenerar artefactos si es necesario.
    """
    print(f"\n{'='*20} INICIANDO CICLO DE ACTUALIZACIÓN COMPLETO {'='*20}")
    
    # Fase 1: Obtener nuevos datos y actualizar el CSV
    new_data_df = update_csv_from_github()
    
    # Fase 2: Actualizar índices y metadatos solo si hubo datos nuevos
    if new_data_df is not None and not new_data_df.empty:
        update_artifacts_incrementally(
            new_data_df=new_data_df,
            models_list=MODELS_TO_PROCESS,
            metadata_json_path=METADATA_FILENAME_GLOBAL
        )
    else:
        print(f"\n[{datetime.now()}] Fase 2 Omitida: No hay datos nuevos, los índices están al día.")
        
    print(f"\n{'='*20} CICLO DE ACTUALIZACIÓN FINALIZADO {'='*20}")

# --- Programador de Tareas ---
if __name__ == "__main__":
    # La función que se programa ahora es la orquestadora principal
    run_complete_pipeline() 
    
    scheduler = BackgroundScheduler(timezone="Europe/Madrid")
    scheduler.add_job(run_complete_pipeline, 'interval', hours=24)
    scheduler.start() # Inicia el scheduler en un hilo secundario
    
    print("\nProgramador iniciado. La próxima actualización se ejecutará en 24 horas.")
    print("El script se mantendrá activo en segundo plano. Presiona Ctrl+C para salir.")
    
    # --- CAMBIO 3: Bucle principal que mantiene vivo el script y es interrumpible ---
    try:
        # Este bucle mantiene el script principal vivo, esperando una interrupción
        while True:
            time.sleep(1) # Duerme por un segundo, lo que permite que Ctrl+C lo interrumpa
    except (KeyboardInterrupt, SystemExit):
        # Cuando se presiona Ctrl+C, se captura la excepción aquí
        print("\nDeteniendo el programador...")
        scheduler.shutdown() # Apaga el scheduler de forma limpia
        print("Programador detenido. Saliendo.")
