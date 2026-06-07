"""
responder.py
-------------------------------------------------------------------
RAG COMPLETO: pregunta -> recuperacion (BigQuery VECTOR_SEARCH)
              -> generacion (Gemini) con CITAS a los articulos.

Flujo:
  1. Embebe la pregunta (text-multilingual-embedding-002, RETRIEVAL_QUERY).
  2. Recupera los top-k chunks mas relevantes desde BigQuery.
  3. Construye un prompt "grounded" y pide a Gemini una respuesta
     basada UNICAMENTE en esos fragmentos, con citas (documento + articulo).

Pensado para areas usuarias, especialistas y operadores del sistema
de compras publicas.

Requisitos:
    pip install google-cloud-bigquery google-genai
    Autenticacion ADC + tabla creada con indexar_bigquery.py.

Uso:
    python responder.py "Que sancion aplica por presentar documentos falsos?"
    python responder.py "plazos de apelacion" --k 6 --modelo gemini-2.5-pro
-------------------------------------------------------------------
"""

import re
import time
import random
import argparse
import threading
import collections
from google.cloud import bigquery
from google import genai
from google.genai.types import EmbedContentConfig, GenerateContentConfig

# ===================== CONFIGURACION =====================
PROJECT_ID = "project-a0134db0-3990-4ec2-bc3"
LOCATION = "us-central1"
BQ_LOCATION = "US"

DATASET = "rag_compras"
TABLA = "chunks_embeddings"

MODELO_EMB = "text-multilingual-embedding-002"
MODELO_GEN = "gemini-2.5-flash"   # alternativa de mayor calidad: gemini-2.5-pro

TOP_K = 5

# GUARDRAIL ESTRICTO (cero alucinaciones) — fuente unica de verdad para el prompt.
# Se inyecta como system_instruction Y se antepone al contexto recuperado (el "candado").
# Texto innegociable: NO modificar sin aprobacion (seguridad juridica).
GUARDRAIL_CONSULTA_GENERAL = (
    "Eres un asistente legal experto en contrataciones públicas. Tu ÚNICA fuente de "
    "verdad es el contexto normativo que se te proporciona a continuación. Tienes "
    "estrictamente prohibido usar tu conocimiento previo o externo. Si la respuesta a "
    "la pregunta no se encuentra explícitamente dentro del contexto proporcionado, "
    "DEBES responder textualmente: \"De acuerdo con el marco normativo actualmente "
    "cargado en el sistema, no dispongo de la información exacta para responder a esta "
    "consulta\". No asumas, no deduzcas plazos y no inventes artículos."
)

# El CLI (responder.py directo) usa el mismo guardrail estricto.
INSTRUCCION_SISTEMA = GUARDRAIL_CONSULTA_GENERAL
# ========================================================

_client = None
_client_lock = threading.Lock()


def cliente():
    """Cliente GenAI PERSISTENTE y THREAD-SAFE (double-checked locking).
    Una sola instancia para todo el proceso: evita la condicion de carrera que, con
    subidas concurrentes (BackgroundTasks en el threadpool), creaba y descartaba
    clientes en paralelo y dejaba el httpx interno 'closed'. El cliente vive lo que
    vive el proceso; NUNCA se cierra al terminar una peticion."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    return _client


def reiniciar_cliente():
    """Recrea el cliente (recuperacion defensiva ante 'client has been closed')."""
    global _client
    with _client_lock:
        _client = None
    return cliente()


# --- Reintentos con BACKOFF EXPONENCIAL + JITTER ante errores transitorios ---
MAX_REINTENTOS = 5               # tras el 1er intento: hasta 5 reintentos
BACKOFF_BASE = 2                 # base de la espera (s): 2, 4, 8, 16, 32...
BACKOFF_MAX = 60                 # tope por espera (s)

_HTTP_NO_REINTENTAR = {400, 401, 403, 404, 405, 409, 422}   # cliente/validacion -> NO reintentar
_HTTP_REINTENTAR = {408, 429, 500, 502, 503, 504}           # tasa/servidor/transitorios -> reintentar


def _codigo_http(e):
    """Extrae el codigo de estado HTTP de una excepcion del SDK (varias convenciones)."""
    for attr in ("code", "status_code", "status"):
        v = getattr(e, attr, None)
        if isinstance(v, int):
            return v
    resp = getattr(e, "response", None)
    v = getattr(resp, "status_code", None)
    return v if isinstance(v, int) else None


def _es_error_de_tasa(e):
    """True si conviene REINTENTAR: 429 (tasa/cuota), 5xx (500/502/503/504), 408 y
    timeouts / errores de conexion transitorios. NO reintenta 400/401/403/404/409/422
    ni errores de validacion (ValueError/TypeError/KeyError)."""
    code = _codigo_http(e)
    if code in _HTTP_NO_REINTENTAR:
        return False
    if code in _HTTP_REINTENTAR:
        return True
    if isinstance(e, (TimeoutError, ConnectionError)):
        return True
    if isinstance(e, (ValueError, TypeError, KeyError)):
        return False
    nombre = type(e).__name__.lower()
    msg = str(e).lower()
    senales = ("resource_exhausted", "too many requests", "rate limit", "quota",
               "unavailable", "overloaded", "deadline", "timeout", "timed out",
               "connection", "reset by peer", "temporarily", "try again",
               "429", "500", "502", "503", "504")
    return any(s in msg for s in senales) or "timeout" in nombre or "connection" in nombre


def _retry_after_segundos(e):
    """Si el error trae Retry-After (header) o retryDelay (cuerpo, ej. 'retryDelay: 7s'),
    devuelve esos segundos para respetarlos en vez del backoff calculado; si no, None."""
    resp = getattr(e, "response", None)
    headers = getattr(resp, "headers", None)
    if headers is not None:
        try:
            ra = headers.get("Retry-After") or headers.get("retry-after")
            if ra is not None:
                return float(ra)
        except (ValueError, TypeError, AttributeError):
            pass
    m = re.search(r'retry[_-]?delay["\s:{]*(?:seconds["\s:]*)?(\d+(?:\.\d+)?)', str(e), re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def con_reintentos(fn, etiqueta="API"):
    """Ejecuta fn() con reintentos en UN SOLO bucle (reutilizable: embeddings y generacion).
    En cada vuelta:
      (a) cliente 'closed' (carrera de concurrencia) -> recrea cliente y reintenta YA;
      (b) 429/5xx/timeout/conexion -> espera Retry-After (si el error lo indica) o
          backoff exponencial con JITTER, con tope BACKOFF_MAX.
    Errores NO reintentables (400/401/403/404/validacion) se propagan de inmediato.
    El time.sleep corre en el hilo de la tarea (threadpool): NO bloquea el event loop."""
    for intento in range(1, MAX_REINTENTOS + 2):     # 1 (inicial) + hasta 5 reintentos
        try:
            return fn()
        except Exception as e:
            # (a) Cliente cerrado bajo concurrencia: recrea y reintenta sin esperar.
            if isinstance(e, RuntimeError) and "closed" in str(e).lower():
                reiniciar_cliente()
                if intento <= MAX_REINTENTOS:
                    continue
                raise
            # (b) Transitorio reintentable.
            if not _es_error_de_tasa(e) or intento > MAX_REINTENTOS:
                raise
            ra = _retry_after_segundos(e)
            if ra is not None:
                espera = min(ra, BACKOFF_MAX)                 # respeta Retry-After del servidor
            else:
                base = min(BACKOFF_BASE ** intento, BACKOFF_MAX)
                espera = base / 2 + random.uniform(0, base / 2)   # equal jitter (50%-100% del base)
            print(f"[{etiqueta}] transitorio ({type(e).__name__}; HTTP {_codigo_http(e)}) "
                  f"intento {intento}/{MAX_REINTENTOS}; espera {espera:.1f}s")
            time.sleep(espera)
    raise RuntimeError(f"{etiqueta}: se agotaron los reintentos.")


def _embed(textos, task_type):
    """Embebe (RETRIEVAL_QUERY/DOCUMENT) con reintentos. Corre en hilo del threadpool:
    el time.sleep del backoff pausa SOLO ese hilo, sin bloquear el event loop ni a otros."""
    cfg = EmbedContentConfig(task_type=task_type)
    return con_reintentos(
        lambda: cliente().models.embed_content(model=MODELO_EMB, contents=textos, config=cfg),
        etiqueta="embeddings")


# --- Control PROACTIVO de tasa para la INDEXACION masiva (embeddings) ---
# Evita el 429 en masa al indexar muchos documentos a la vez. SOLO aplica a la
# indexacion (embeber_para_indexar); las CONSULTAS interactivas no se throttlean.
EMBED_CONCURRENCIA_MAX = 4       # llamadas de embeddings simultaneas (indexacion)
EMBED_RPM_MAX = 100              # tope de peticiones por minuto (0 = sin limite RPM)

_embed_sem = threading.Semaphore(EMBED_CONCURRENCIA_MAX)
_rpm_lock = threading.Lock()
_rpm_marcas = collections.deque()


def _limitar_tasa_embed():
    """Espera (en el hilo worker) hasta respetar EMBED_RPM_MAX peticiones/min (ventana deslizante)."""
    if not EMBED_RPM_MAX:
        return
    while True:
        with _rpm_lock:
            ahora = time.monotonic()
            while _rpm_marcas and ahora - _rpm_marcas[0] > 60:
                _rpm_marcas.popleft()
            if len(_rpm_marcas) < EMBED_RPM_MAX:
                _rpm_marcas.append(ahora)
                return
            espera = 60 - (ahora - _rpm_marcas[0]) + 0.01
        time.sleep(min(espera, 5))   # libera el lock mientras espera


def _embed_indexar(textos):
    """Embed para INDEXACION con control proactivo: tope de concurrencia (semaforo) +
    tope de peticiones/min. El backoff reactivo (con_reintentos, dentro de _embed) sigue
    como red de seguridad si aun asi se cuela un 429."""
    with _embed_sem:                 # limita cuantos hilos embeben a la vez
        _limitar_tasa_embed()        # limita el ritmo (RPM)
        return _embed(textos, "RETRIEVAL_DOCUMENT")


# Identificador UNICO de cada norma: prefijo bib_<id> (Biblioteca) o documento (corpus).
# Misma expresion en el listado (/api/normas) y en el pre-filtro -> seleccion individual.
_DOC_ID_SQL = r"IFNULL(REGEXP_EXTRACT(chunk_id, r'^(bib_[0-9a-f]+)__'), documento)"


def _construir_prefiltro(filtros):
    """Traduce los filtros del usuario en una clausula WHERE parametrizada para el
    PRE-FILTERING (se aplica ANTES de la busqueda vectorial). Devuelve (where|None, params)."""
    if not filtros:
        return None, []
    conds, params = [], []

    cats = [c for c in (filtros.get("categorias") or []) if c]
    if cats:
        conds.append("categoria IN UNNEST(@f_cats)")
        params.append(bigquery.ArrayQueryParameter("f_cats", "STRING", cats))

    if filtros.get("excluir_derogada"):
        conds.append("IFNULL(vigente, TRUE) = TRUE")

    anio = str(filtros.get("anio") or "Todos")
    if anio == "Anteriores":
        conds.append("anio < 2024")
    elif anio.isdigit():
        conds.append("anio = @f_anio")
        params.append(bigquery.ScalarQueryParameter("f_anio", "INT64", int(anio)))

    # Seleccion INDIVIDUAL de normas (convive con los filtros de arriba, en AND).
    # None = sin restriccion individual; lista vacia = ninguna norma seleccionada.
    normas = filtros.get("normas")
    if normas is not None:
        normas = [n for n in normas if n]
        if normas:
            conds.append(f"{_DOC_ID_SQL} IN UNNEST(@f_normas)")
            params.append(bigquery.ArrayQueryParameter("f_normas", "STRING", normas))
        else:
            conds.append("1 = 0")   # seleccion vacia explicita -> no busca en ninguna norma

    return (" AND ".join(conds) if conds else None), params


def listar_normas():
    """Lista las normas DISTINTAS del vector store (corpus + Biblioteca), cada una con su
    doc_id unico, etiqueta y metadatos. Alimenta el menu de seleccion individual."""
    bq = bigquery.Client(project=PROJECT_ID)
    sql = f"""
    SELECT
      {_DOC_ID_SQL} AS doc_id,
      ANY_VALUE(documento) AS documento,
      ANY_VALUE(categoria) AS categoria,
      MAX(anio) AS anio,
      LOGICAL_AND(IFNULL(vigente, TRUE)) AS vigente,
      COUNT(*) AS n_chunks
    FROM `{PROJECT_ID}.{DATASET}.{TABLA}`
    GROUP BY doc_id
    ORDER BY categoria, documento
    """
    out = []
    for r in bq.query(sql, location=BQ_LOCATION).result():
        did = r["doc_id"] or ""
        out.append({
            "doc_id": did,
            "label": doc_label(r["documento"], did),
            "categoria": r["categoria"],
            "anio": r["anio"],
            "vigente": bool(r["vigente"]) if r["vigente"] is not None else True,
            "n_chunks": r["n_chunks"],
            "fuente": "biblioteca" if did.startswith("bib_") else "corpus",
        })
    return out


def recuperar(pregunta, k, filtros=None):
    """Devuelve los top-k chunks relevantes desde BigQuery.
    Busqueda HIBRIDA: si se pasan `filtros`, primero se descartan por metadatos
    (categoria / vigencia / anio) y solo luego se calcula la similitud (coseno)."""
    qemb = _embed([pregunta], "RETRIEVAL_QUERY").embeddings[0].values

    bq = bigquery.Client(project=PROJECT_ID)
    tabla = f"`{PROJECT_ID}.{DATASET}.{TABLA}`"
    where, fparams = _construir_prefiltro(filtros)
    # PRE-FILTERING: VECTOR_SEARCH busca solo dentro del subconjunto que cumple los metadatos.
    relacion = f"(SELECT * FROM {tabla} WHERE {where})" if where else f"TABLE {tabla}"

    sql = f"""
    SELECT base.chunk_id AS chunk_id, base.documento AS documento, base.articulo_num AS articulo_num,
           base.articulo_titulo AS articulo_titulo, base.texto AS texto, distance
    FROM VECTOR_SEARCH(
      {relacion}, 'embedding',
      (SELECT @qemb AS embedding),
      top_k => @k, distance_type => 'COSINE')
    ORDER BY distance
    """
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("qemb", "FLOAT64", qemb),
        bigquery.ScalarQueryParameter("k", "INT64", k),
        *fparams,
    ])
    return list(bq.query(sql, job_config=cfg, location=BQ_LOCATION).result())


def embeber_para_indexar(textos, lote=16):
    """Embebe una lista de fragmentos para INDEXAR (task_type=RETRIEVAL_DOCUMENT,
    simetrico al RETRIEVAL_QUERY de la consulta). Devuelve lista de vectores 768-dim."""
    vects = []
    for i in range(0, len(textos), lote):
        sub = textos[i:i + lote]
        resp = _embed_indexar(sub)   # control proactivo de tasa (concurrencia + RPM)
        vects.extend([e.values for e in resp.embeddings])
    return vects


# Esquema de la tabla vectorial (debe coincidir con indexar_bigquery.SCHEMA).
_SCHEMA_VECTOR = [
    bigquery.SchemaField("chunk_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("categoria", "STRING"),
    bigquery.SchemaField("documento", "STRING"),
    bigquery.SchemaField("articulo_num", "STRING"),
    bigquery.SchemaField("articulo_titulo", "STRING"),
    bigquery.SchemaField("parte", "INTEGER"),
    bigquery.SchemaField("n_chars", "INTEGER"),
    bigquery.SchemaField("texto", "STRING"),
    bigquery.SchemaField("anio", "INTEGER"),
    bigquery.SchemaField("vigente", "BOOLEAN"),
    bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
]


def indexar_chunks(filas):
    """Inserta (WRITE_APPEND) filas con embedding+metadatos en la tabla vectorial."""
    bq = bigquery.Client(project=PROJECT_ID)
    tabla_id = f"{PROJECT_ID}.{DATASET}.{TABLA}"
    job = bq.load_table_from_json(
        filas, tabla_id,
        job_config=bigquery.LoadJobConfig(
            schema=_SCHEMA_VECTOR,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        ),
    )
    job.result()
    return len(filas)


def eliminar_por_prefijo(prefijo_chunk_id):
    """Borra del vector store todos los chunks cuyo chunk_id empiece por el prefijo
    (usado para eliminar un documento institucional completo)."""
    bq = bigquery.Client(project=PROJECT_ID)
    sql = f"DELETE FROM `{PROJECT_ID}.{DATASET}.{TABLA}` WHERE STARTS_WITH(chunk_id, @p)"
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("p", "STRING", prefijo_chunk_id)])
    bq.query(sql, job_config=cfg, location=BQ_LOCATION).result()


def doc_label(documento, chunk_id=None):
    """Etiqueta legible y UNICA de un documento del vector store (FUENTE DE VERDAD
    compartida: la usan responder.py y app.py).
    Los documentos de la Biblioteca (chunk_id 'bib_<id>__...') usan SIEMPRE su propio
    nombre: asi una directiva cuyo nombre contiene 'reglamento'/'ley-general' no se
    etiqueta (ni se cita) erroneamente como 'Reglamento'/'Ley'. El mapeo Ley/Reglamento
    solo aplica al corpus normativo."""
    d = documento or ""
    if chunk_id and str(chunk_id).startswith("bib_"):
        return d or "Documento"
    if "ley-general" in d:
        return "Ley"
    if "reglamento" in d:
        return "Reglamento"
    return d or "Documento"


def construir_contexto(filas):
    """Arma el bloque de contexto con los fragmentos numerados."""
    bloques = []
    for i, f in enumerate(filas, start=1):
        doc = doc_label(f["documento"], f.get("chunk_id"))
        texto = " ".join(f["texto"].split())
        bloques.append(
            f"[Fragmento {i}] ({doc}, Art. {f['articulo_num']} - {f['articulo_titulo']})\n{texto}"
        )
    return "\n\n".join(bloques)


def responder(pregunta, k=TOP_K, modelo=MODELO_GEN):
    filas = recuperar(pregunta, k)
    if not filas:
        return "No se encontraron fragmentos normativos relevantes.", []

    contexto = construir_contexto(filas)
    # EL CANDADO: el guardrail va como system_instruction Y antepuesto al contexto
    # (consistente con la web). Mismo texto innegociable.
    prompt = (
        GUARDRAIL_CONSULTA_GENERAL + "\n\n"
        f"FRAGMENTOS NORMATIVOS:\n{contexto}\n\n"
        f"PREGUNTA DEL USUARIO:\n{pregunta}\n\n"
        f"Redacta la respuesta siguiendo las reglas, citando los articulos."
    )
    resp = con_reintentos(
        lambda: cliente().models.generate_content(
            model=modelo,
            contents=prompt,
            config=GenerateContentConfig(
                system_instruction=INSTRUCCION_SISTEMA,
                temperature=0.2,
            ),
        ),
        etiqueta="generacion")
    return resp.text, filas


def main():
    parser = argparse.ArgumentParser(description="RAG completo (pregunta -> respuesta con citas).")
    parser.add_argument("pregunta", help="Pregunta en lenguaje natural.")
    parser.add_argument("--k", type=int, default=TOP_K, help="Fragmentos a recuperar (default 5).")
    parser.add_argument("--modelo", default=MODELO_GEN, help="Modelo Gemini (default gemini-2.5-flash).")
    args = parser.parse_args()

    print("=" * 70)
    print(f" PREGUNTA: {args.pregunta}")
    print("=" * 70)

    texto, filas = responder(args.pregunta, args.k, args.modelo)

    print("\n--- RESPUESTA ---\n")
    print(texto)
    print("\n--- FUENTES RECUPERADAS ---")
    for i, f in enumerate(filas, start=1):
        doc = doc_label(f["documento"], f.get("chunk_id"))
        print(f"  [{i}] {doc}, Art. {f['articulo_num']}: {f['articulo_titulo']} "
              f"(coseno {1 - f['distance']:.3f})")


if __name__ == "__main__":
    main()
