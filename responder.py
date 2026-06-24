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
from google.genai.types import EmbedContentConfig, GenerateContentConfig, Content, Part
# Config CENTRAL de embeddings (mismo modelo/dims que generar_embeddings.py).
from config_emb import MODELO_EMB, OUTPUT_DIM, recortar, normalizar
# Chequeo de fidelidad de citas (determinista, sin LLM).
from citas import verificar_citas

# ===================== CONFIGURACION =====================
PROJECT_ID = "project-a0134db0-3990-4ec2-bc3"
LOCATION = "us-central1"
BQ_LOCATION = "US"

DATASET = "rag_compras"
TABLA = "chunks_embeddings"

MODELO_GEN = "gemini-2.5-flash"   # alternativa de mayor calidad: gemini-2.5-pro
MODELO_EXPANSION = "gemini-2.5-flash"   # reformulacion de consulta (rapido/barato)

TOP_K = 10                        # fragmentos a recuperar (configurable; antes 5)

# Mejora de recuperacion (todo configurable):
EXPANDIR_CONSULTA = True          # reformula/expande la consulta (tema + sinonimos)
COMPLETAR_ARTICULO = True         # trae todas las partes del mismo articulo/numeral
MAX_PARTES_ARTICULO = 12          # tope de partes por articulo (evita inflar el contexto)

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
    """Embebe (RETRIEVAL_QUERY/DOCUMENT) con reintentos y devuelve vectores YA NORMALIZADOS
    (L2) al OUTPUT_DIM configurado. Corre en hilo del threadpool: el time.sleep del backoff
    pausa SOLO ese hilo, sin bloquear el event loop ni a otros."""
    cfg = EmbedContentConfig(task_type=task_type, output_dimensionality=OUTPUT_DIM)
    entradas = [recortar(t) for t in textos]
    resp = con_reintentos(
        lambda: cliente().models.embed_content(model=MODELO_EMB, contents=entradas, config=cfg),
        etiqueta="embeddings")
    return [normalizar(e.values) for e in resp.embeddings]


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

    # CATEGORIAS TRANSVERSALES: lista blanca (Lectura A "marcado = se busca"). Se elimina la
    # regla "sin seleccion = todas": una categoria desmarcada NO se busca. Las resoluciones del
    # Tribunal NO son un checkbox aqui (se gobiernan por sus toggles de subtipo, mas abajo), por
    # eso quedan EXENTAS de esta lista blanca: pasan este gate y la condicion de subtipo decide
    # si entran. Sin ninguna categoria marcada, solo pueden entrar resoluciones (via sus toggles).
    cats = [c for c in (filtros.get("categorias") or []) if c]
    if cats:
        conds.append("(categoria IN UNNEST(@f_cats) OR categoria = 'resoluciones_tribunal')")
        params.append(bigquery.ArrayQueryParameter("f_cats", "STRING", cats))
    else:
        conds.append("categoria = 'resoluciones_tribunal'")

    # RESOLUCIONES DEL TRIBUNAL: control FINO por subtipo (toggles independientes). Solo
    # entran los subtipos activados; 'otra' nunca (por ahora, sin control propio). Ambos OFF
    # = ninguna resolucion (igual que antes). NO afecta a las demas categorias (transversales,
    # siempre elegibles). El PESO por jerarquia (ranking) es aparte: decide cuanto pesan, no
    # cuales entran.
    subtipos_ok = []
    if filtros.get("incluir_apelacion"):
        subtipos_ok.append("apelacion")
    if filtros.get("incluir_sancionadoras"):
        subtipos_ok.append("sancionadora")
    if subtipos_ok:
        conds.append("(IFNULL(categoria, '') != 'resoluciones_tribunal' "
                     "OR subtipo IN UNNEST(@f_subtipos))")
        params.append(bigquery.ArrayQueryParameter("f_subtipos", "STRING", subtipos_ok))
    else:
        conds.append("IFNULL(categoria, '') != 'resoluciones_tribunal'")

    if filtros.get("excluir_derogada"):
        conds.append("IFNULL(vigente, TRUE) = TRUE")

    anio = str(filtros.get("anio") or "Todos")
    if anio == "Anteriores":
        conds.append("anio < 2024")
    elif anio.isdigit():
        conds.append("anio = @f_anio")
        params.append(bigquery.ScalarQueryParameter("f_anio", "INT64", int(anio)))

    # Refinamiento INDIVIDUAL de normas (convive con los filtros de arriba, en AND). Aplica solo
    # a normativa TRANSVERSAL: las resoluciones no tienen lista por-documento (son ~9.940) y se
    # gobiernan por sus toggles de subtipo, por eso quedan exentas (un refinamiento de documentos
    # transversales no debe apagar las resoluciones encendidas). None = sin restriccion individual;
    # lista vacia = ninguna norma transversal -> solo pueden entrar resoluciones (segun toggles).
    normas = filtros.get("normas")
    if normas is not None:
        normas = [n for n in normas if n]
        if normas:
            conds.append(f"({_DOC_ID_SQL} IN UNNEST(@f_normas) "
                         "OR categoria = 'resoluciones_tribunal')")
            params.append(bigquery.ArrayQueryParameter("f_normas", "STRING", normas))
        else:
            conds.append("categoria = 'resoluciones_tribunal'")

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
            "documento": r["documento"],                 # slug crudo (para derivar el nombre)
            "label": doc_label(r["documento"], did),     # etiqueta base (el endpoint la re-resuelve)
            "categoria": r["categoria"],
            "anio": r["anio"],
            "vigente": bool(r["vigente"]) if r["vigente"] is not None else True,
            "n_chunks": r["n_chunks"],
            "fuente": "biblioteca" if did.startswith("bib_") else "corpus",
        })
    return out


_COLS_CHUNK = ("chunk_id", "categoria", "documento", "tipo_referencia", "referencia", "fase",
               "emisor", "articulo_num", "articulo_titulo", "parte", "texto")

# ===== JERARQUIA DE AUTORIDAD (un solo lugar) =====
# Orden mayor -> menor autoridad. Se traduce en una penalidad ADITIVA sobre la distancia
# coseno: una norma de mayor autoridad NO es desplazada por una de menor autoridad con
# similitud parecida. El Reglamento (leyes_y_reglamentos) flota por encima de resoluciones
# igual de "parecidas".
ORDEN_AUTORIDAD = ("leyes_y_reglamentos", "directivas", "opiniones",
                   "resoluciones_tribunal", "documentos_orientacion")
PESO_AUTORIDAD = {cat: round(0.05 * i, 3) for i, cat in enumerate(ORDEN_AUTORIDAD)}
PESO_AUTORIDAD_DEFAULT = 0.10      # categoria desconocida/None


def _score_autoridad(fila):
    """Score de ranking: distancia coseno + penalidad por (baja) autoridad de categoria.
    Menor = mejor."""
    dist = fila.get("distance")
    dist = 1.0 if dist is None else dist
    return dist + PESO_AUTORIDAD.get(fila.get("categoria"), PESO_AUTORIDAD_DEFAULT)


def _buscar(consulta, k, filtros=None):
    """Una busqueda vectorial (top-k) -> lista de dicts. Busqueda HIBRIDA: si se pasan
    `filtros`, primero se descartan por metadatos (categoria/vigencia/anio) y solo luego
    se calcula la similitud (coseno)."""
    qemb = _embed([consulta], "RETRIEVAL_QUERY")[0]
    bq = bigquery.Client(project=PROJECT_ID)
    tabla = f"`{PROJECT_ID}.{DATASET}.{TABLA}`"
    where, fparams = _construir_prefiltro(filtros)
    relacion = f"(SELECT * FROM {tabla} WHERE {where})" if where else f"TABLE {tabla}"
    sql = f"""
    SELECT base.chunk_id AS chunk_id, base.categoria AS categoria, base.documento AS documento,
           base.tipo_referencia AS tipo_referencia, base.referencia AS referencia,
           base.fase AS fase, base.emisor AS emisor,
           base.articulo_num AS articulo_num, base.articulo_titulo AS articulo_titulo,
           base.parte AS parte, base.texto AS texto, distance
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
    return [dict(r) for r in bq.query(sql, job_config=cfg, location=BQ_LOCATION).result()]


def _completar_articulos(filas):
    """Para cada chunk recuperado que sea articulo/numeral, trae TODAS las partes del
    mismo (documento + referencia) y las ensambla CONTIGUAS y en orden de `parte`, para
    darle al modelo el articulo COMPLETO en vez de un fragmento. Dedup por chunk_id."""
    sep = "\x01"
    pares = {f'{f.get("documento")}{sep}{f.get("referencia")}'
             for f in filas
             if f.get("tipo_referencia") in ("articulo", "numeral")
             and f.get("referencia") and f.get("documento")}
    if not pares:
        return filas

    bq = bigquery.Client(project=PROJECT_ID)
    tabla = f"`{PROJECT_ID}.{DATASET}.{TABLA}`"
    sql = f"""
    SELECT {", ".join(_COLS_CHUNK)}
    FROM {tabla}
    WHERE CONCAT(documento, '{sep}', referencia) IN UNNEST(@claves)
      AND tipo_referencia IN ('articulo', 'numeral')
    ORDER BY documento, referencia, parte
    """
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("claves", "STRING", sorted(pares))])
    porart = collections.defaultdict(list)
    for r in bq.query(sql, job_config=cfg, location=BQ_LOCATION).result():
        porart[f'{r["documento"]}{sep}{r["referencia"]}'].append(dict(r))

    salida, vistos, emitidos = [], set(), set()
    for f in filas:
        clave = f'{f.get("documento")}{sep}{f.get("referencia")}'
        if clave in porart and clave not in emitidos:
            emitidos.add(clave)
            for parte in porart[clave][:MAX_PARTES_ARTICULO]:
                if parte["chunk_id"] in vistos:
                    continue
                vistos.add(parte["chunk_id"])
                parte.setdefault("distance", f.get("distance"))   # ordena junto al recuperado
                salida.append(parte)
        elif f.get("chunk_id") not in vistos:
            vistos.add(f.get("chunk_id"))
            salida.append(f)
    return salida


def recuperar(consulta, k=TOP_K, filtros=None, completar=COMPLETAR_ARTICULO):
    """Recupera chunks relevantes. `consulta` puede ser un str o una LISTA de consultas
    (p.ej. [original, expandida]); se busca cada una y se FUSIONAN sin duplicar (mejor
    distancia), garantizando recall. Si `completar`, ademas trae el articulo completo."""
    consultas = [consulta] if isinstance(consulta, str) else list(consulta)
    consultas = list(dict.fromkeys(c.strip() for c in consultas if c and c.strip()))
    if not consultas:
        return []

    fusion = {}
    for c in consultas:
        for f in _buscar(c, k, filtros):
            cid = f["chunk_id"]
            if cid not in fusion or f["distance"] < fusion[cid]["distance"]:
                fusion[cid] = f
    # Ranking por AUTORIDAD: distancia + penalidad de categoria (jerarquia de normas).
    filas = sorted(fusion.values(), key=_score_autoridad)[:k]
    return _completar_articulos(filas) if completar else filas


# ===== Expansion/reformulacion de consulta (mejor recall en follow-ups e imprecisas) =====
try:                                   # apaga el "thinking" del flash si esta disponible
    from google.genai.types import ThinkingConfig
    _THINK_OFF = ThinkingConfig(thinking_budget=0)
except Exception:
    _THINK_OFF = None

_RE_ESPECIFICA = re.compile(
    r'\b(art|articulo|art[ií]culo|numeral|inciso|ley|reglamento|opini[oó]n|directiva|'
    r'decreto|constituci[oó]n|tuo|c[oó]digo)\b', re.IGNORECASE)


def _es_especifica(pregunta):
    """True si la consulta ya es concreta/autocontenida (numero de art., norma citada o
    suficientes terminos de contenido): sin historial, no necesita expansion."""
    p = pregunta or ""
    if any(ch.isdigit() for ch in p) or _RE_ESPECIFICA.search(p):
        return True
    return len([w for w in re.findall(r'\w+', p) if len(w) > 3]) >= 6


_INSTR_EXPANSION = (
    "Eres un asistente de busqueda juridica. Dada la conversacion y la nueva consulta, "
    "devuelve UNICAMENTE una lista corta (5 a 12 palabras) de terminos TEMATICOS y "
    "SINONIMOS en espanol que ayuden a encontrar la norma por el TEMA. Reglas: (a) NO "
    "repitas ni reescribas la consulta; solo agrega terminos nuevos. (b) No inventes "
    "numeros de articulo ni nombres de norma que no aparezcan. (c) Si es un follow-up "
    "(p.ej. 'y para obras?'), agrega el tema de los turnos previos. (d) Si la consulta ya "
    "es clara y no aportarias nada util, responde con una linea vacia. Responde solo los "
    "terminos, sin prefijos ni comillas.")


def reformular_consulta(pregunta, historial_texto=""):
    """Devuelve SOLO terminos a AÑADIR (sinonimos/tema); '' si no aporta. NO reescribe ni
    quita: la consulta final sera 'original + estos terminos', asi los terminos especificos
    del original (numeros de articulo, nombres, keywords) NUNCA se pierden."""
    prompt = (f"Conversacion reciente:\n{historial_texto or '(ninguna)'}\n\n"
              f"Nueva consulta: {pregunta}\n\n{_INSTR_EXPANSION}")
    cfg = dict(temperature=0.1, max_output_tokens=256)
    if _THINK_OFF is not None:
        cfg["thinking_config"] = _THINK_OFF
    try:
        resp = con_reintentos(
            lambda: cliente().models.generate_content(
                model=MODELO_EXPANSION, contents=prompt,
                config=GenerateContentConfig(**cfg)),
            etiqueta="expansion")
        extra = " ".join((resp.text or "").split()).strip(' "\'')
        return extra if 0 < len(extra) <= 200 else ""
    except Exception:
        return ""                      # ante fallo, NO degradar: se busca con el original


def consultas_busqueda(pregunta, historial_texto=""):
    """Consultas para `recuperar`: [original] o [original, 'original + expansion'].
    Salta la expansion (ahorra la llamada flash) si NO hay historial y la consulta ya es
    especifica. El dual-search preserva el recall de la consulta directa."""
    pregunta = (pregunta or "").strip()
    if not EXPANDIR_CONSULTA or not pregunta:
        return [pregunta] if pregunta else []
    if not historial_texto and _es_especifica(pregunta):
        return [pregunta]
    extra = reformular_consulta(pregunta, historial_texto)
    return [pregunta, f"{pregunta} {extra}"] if extra else [pregunta]


def embeber_para_indexar(textos, lote=1):   # gemini-embedding-001: 1 texto por request
    """Embebe una lista de fragmentos para INDEXAR (task_type=RETRIEVAL_DOCUMENT,
    simetrico al RETRIEVAL_QUERY de la consulta). Devuelve vectores normalizados al
    OUTPUT_DIM configurado (mismo modelo/dims que la consulta -> coseno consistente)."""
    vects = []
    for i in range(0, len(textos), lote):
        sub = textos[i:i + lote]
        vects.extend(_embed_indexar(sub))   # control proactivo de tasa; ya normaliza
    return vects


# Esquema de la tabla vectorial (debe coincidir con indexar_bigquery.SCHEMA y la tabla viva).
_SCHEMA_VECTOR = [
    bigquery.SchemaField("chunk_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("categoria", "STRING"),
    bigquery.SchemaField("documento", "STRING"),
    bigquery.SchemaField("tipo_referencia", "STRING"),
    bigquery.SchemaField("referencia", "STRING"),
    bigquery.SchemaField("articulo_num", "STRING"),
    bigquery.SchemaField("articulo_titulo", "STRING"),
    bigquery.SchemaField("parte", "INTEGER"),
    bigquery.SchemaField("n_chars", "INTEGER"),
    bigquery.SchemaField("texto", "STRING"),
    bigquery.SchemaField("fase", "STRING"),
    bigquery.SchemaField("emisor", "STRING"),
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


_RE_ID_PREFIJO = re.compile(r'^\d{3,}-(?:ref-)?', re.IGNORECASE)
_ACRONIMOS = {"tuo", "dl", "ds", "oece", "dtn", "ef", "mef", "pac", "jprd", "tup", "ruc", "uit"}


def _titulo_legible(documento):
    """Convierte el nombre de archivo en un titulo legible: quita el id numerico inicial,
    cambia '-'/'_' por espacios y pone en mayuscula siglas conocidas (TUO, DL, OECE...)."""
    s = _RE_ID_PREFIJO.sub('', documento or '')
    s = re.sub(r'\s+', ' ', s.replace('_', ' ').replace('-', ' ')).strip()
    if not s:
        return "Documento"
    s = ' '.join(w.upper() if w.lower() in _ACRONIMOS else w for w in s.split(' '))
    return (s[0].upper() + s[1:])[:80].strip()


def doc_label(documento, chunk_id=None):
    """Etiqueta legible y UNICA de un documento del vector store (FUENTE DE VERDAD
    compartida: la usan responder.py y app.py).
    Solo los DOS pilares de contrataciones reciben etiqueta corta fija ('Ley 32069',
    'Reglamento'). Cualquier otro documento usa un TITULO LEGIBLE derivado de su nombre,
    asi una directiva cuyo nombre contiene 'reglamento' NO se etiqueta como 'Reglamento'."""
    d = documento or ""
    dl = d.lower()
    if chunk_id and str(chunk_id).startswith("bib_"):
        return _titulo_legible(d)
    if "ley-general-de-contrataciones" in dl:
        return "Ley 32069"
    if "reglamento-de-la-ley-de-contrataciones" in dl:
        return "Reglamento"
    return _titulo_legible(d)


def formato_cita(fila, nombre=None):
    """Cita NATURAL segun el tipo de referencia (reemplaza el viejo 'Art. X' para todo).
    Si la fila no tiene tipo_referencia/referencia (corpus aun no recargado), cae al
    esquema anterior basado en articulo_num.
    `nombre`: si se pasa, se usa como nombre del documento en vez de doc_label (capa de
    presentacion: nombre_mostrar resuelto). Permite re-armar la cita con el nombre editado."""
    get = fila.get if hasattr(fila, "get") else (lambda k, d=None: fila[k] if k in fila else d)
    doc = nombre if nombre else doc_label(get("documento"), get("chunk_id"))
    tipo = get("tipo_referencia")
    ref = get("referencia")
    if not ref:
        an = get("articulo_num")
        return f"{doc}, Art. {an}" if an not in (None, "", "0") else doc
    if tipo == "articulo":
        return f"{doc}, Art. {ref}"
    if tipo == "numeral":
        return f"{doc}, Numeral {ref}"
    if tipo == "opinion":
        return f"Opinión {ref}"
    if tipo == "considerando":
        return f"{doc}, {ref}"          # ref ya = 'Fundamento 7' / 'Antecedente 3' / 'Resuelve 1'
    if tipo == "anexo":
        return f"{doc}, {ref}"          # ref = 'Anexo 2' / 'Formato 6'
    return f"{doc}, {ref}"              # seccion u otros


# ======================= NOMBRE LEGIBLE (capa de presentacion) =======================
# resolver_nombre(documento) por prioridad: override (lo maneja app.py, con SQLite) ->
# nombre_derivado(): mapa curado de singulares -> derivacion por tipo -> fallback legible.
# Es PRESENTACION pura: no toca embeddings/recuperacion/jerarquia/subtipo. Corre EN VIVO en
# cada render (lista de Normas y citas), asi cualquier documento NUEVO se auto-etiqueta sin
# intervencion, siempre que su dato base (numero, etc.) este en `documento`.

_RE_DOC_ID_BIB = re.compile(r'^(bib_[0-9a-f]+)__')


def doc_id_de(documento, chunk_id=None):
    """Identificador UNICO de documento (espejo de _DOC_ID_SQL): para chunks de Biblioteca,
    el prefijo bib_<hash> del chunk_id; para el corpus, el propio `documento`. Es la clave de
    override y la que enlaza una cita con su norma para propagar renombrados."""
    m = _RE_DOC_ID_BIB.match(str(chunk_id or ""))
    return m.group(1) if m else documento


# --- Mapa curado de singulares: (predicado sobre documento.lower()) -> nombre oficial. ---
def _es_ley_general(dl):
    return "ley-general-de-contrataciones" in dl and "reglamento" not in dl

_CURADO_SINGULARES = [
    (_es_ley_general, "Ley N° 32069 – Ley General de Contrataciones Públicas"),
    (lambda dl: "reglamento-de-la-ley-de-contrataciones" in dl or "ref-reglamento" in dl,
     "Reglamento de la Ley N° 32069 (DS 009-2025-EF)"),
    (lambda dl: "30225" in dl, "Ley N° 30225 – Ley de Contrataciones del Estado (TUO)"),
    (lambda dl: "344-2018" in dl, "Reglamento de la Ley N° 30225 (DS 344-2018-EF)"),
    (lambda dl: "27444" in dl, "Ley N° 27444 – Ley del Procedimiento Administrativo General (TUO)"),
    (lambda dl: "codigo-civil" in dl, "Código Civil"),
]


def _numero_resolucion(documento):
    """Extrae el numero completo de una resolucion desde `documento` (ya trae el numero).
    Tolera el formato canonico (resolucion-n-8736-2025-tcp-s5) y los irregulares
    (0437-2026, 04813-2026-tcp-s2). Devuelve p.ej. '8736-2025-TCP-S5' o None si no hay numero."""
    post = _RE_ID_PREFIJO.sub('', documento or '').lower()
    m = re.search(r'resoluci[oó]n[\s_\-]*n?[º°\.\s_\-]*([0-9].*)$', post)
    core = (m.group(1) if m else post).strip(' -_')
    if not re.match(r'^\d', core):          # no parece un numero de resolucion
        return None
    core = re.sub(r'[\s_]+', '-', core)
    core = re.sub(r'-{2,}', '-', core).strip('-')
    return core.upper()


def _numero_opinion(documento):
    """Numero y emisor de una opinion desde `documento` (opinion-d042-2025-oece-dtn) ->
    ('D042-2025', 'OECE'). Devuelve (numero|None, emisor|'')."""
    post = _RE_ID_PREFIJO.sub('', documento or '').lower()
    m = re.search(r'opini[oó]n[\s_\-]*n?[º°\.\s_\-]*([a-z]?\d+[\-/]\d{4})', post)
    num = m.group(1).upper().replace('/', '-') if m else None
    emisor = "OECE" if "oece" in post else ("DGA" if "dga" in post else "")
    return num, emisor


def nombre_derivado(documento, categoria=None, numero=None):
    """Nombre legible SIN overrides (puro): curado -> derivacion por tipo -> fallback legible.
    `numero`: si se provee (no usado hoy; reservado por si se persiste), tiene prioridad sobre
    la extraccion desde `documento`."""
    dl = (documento or "").lower()
    for pred, nombre in _CURADO_SINGULARES:
        if pred(dl):
            return nombre
    if categoria == "resoluciones_tribunal":
        num = numero or _numero_resolucion(documento)
        if num:
            return f"Resolución N° {num}"
    elif categoria == "opiniones":
        num, emisor = _numero_opinion(documento)
        if num:
            return f"Opinión N° {num}" + (f"/{emisor}" if emisor else "")
    # directivas (sin numero en `documento`), documentos_orientacion y cualquier otro:
    # titulo legible derivado del nombre de archivo.
    return _titulo_legible(documento)


def construir_contexto(filas):
    """Arma el bloque de contexto con los fragmentos numerados y su cita natural."""
    bloques = []
    for i, f in enumerate(filas, start=1):
        texto = " ".join(f["texto"].split())
        bloques.append(f"[Fragmento {i}] ({formato_cita(f)})\n{texto}")
    return "\n\n".join(bloques)


# ===================================================================
# GENERACION UNIFICADA — fuente unica compartida por el endpoint (/api/chat),
# el CLI (responder()) y el harness (eval/). Movida desde app.py para que el
# eval mida EXACTAMENTE el prompt que se envia a produccion.
# ===================================================================
MAX_TURNOS_HISTORIAL = 12          # turnos de historial inyectados al modelo

# Instruccion de formato de CITAS (se AÑADE al prompt; no reemplaza el guardrail).
INSTRUCCION_CITAS = (
    "FORMATO DE CITAS: tras cada afirmacion, coloca el marcador [N] (por ejemplo [1], [2]) "
    "del/los fragmento(s) de NORMAS RECUPERADAS que la respaldan. Usa solo numeros de "
    "fragmentos existentes; no inventes marcadores."
)

# Instruccion de PRECISION (se AÑADE; NO modifica el texto anti-alucinaciones del guardrail).
# Evita el "no dispongo" falso cuando la cita del usuario es imprecisa pero el contexto SI
# contiene la norma: en ese caso responde y aclara la referencia correcta.
INSTRUCCION_PRECISION = (
    "PRECISION Y ALCANCE: responde UNICAMENTE con la informacion del contexto. Si el usuario "
    "cita un articulo o una norma de forma imprecisa (por ejemplo, atribuye un tema al "
    "Reglamento cuando el contexto lo regula en la Ley, o viceversa) pero el contexto SI "
    "contiene la norma pertinente, RESPONDE con base en el contexto y ACLARA la referencia "
    "correcta (que norma y articulo lo regulan). Recurre a la frase de que no dispones de la "
    "informacion SOLO cuando el contexto realmente no la contenga."
)

# Instruccion de ESTRUCTURA de la respuesta (ADITIVA; el guardrail no se toca).
INSTRUCCION_ESTRUCTURA = (
    "ESTRUCTURA DE LA RESPUESTA: comienza con una apertura DIRECTA de 1-2 frases que responda "
    "la pregunta. Si hay varias reglas, supuestos o condiciones, desarrollalas despues en "
    "puntos o numeracion (una idea por punto). Cierra indicando la referencia normativa "
    "principal que sustenta la respuesta."
)

# Instruccion de CRUCE Ley<->Reglamento ANCLADO al contexto (ADITIVA).
INSTRUCCION_CRUCE = (
    "CRUCE LEY-REGLAMENTO (OBLIGATORIO): revisa TODOS los fragmentos del contexto. Si ademas "
    "del articulo de la Ley que responde la pregunta hay articulos del Reglamento (u otras "
    "normas) que desarrollan ese mismo tema, DEBES mencionarlos en la respuesta, indicando la "
    "relacion explicitamente (por ejemplo: 'regulado en el art. X de la Ley [n] y desarrollado "
    "en los arts. Y [n] y Z [n] del Reglamento') y citando cada uno con su marcador [N]. "
    "REGLA DURA: solo puedes cruzar normas PRESENTES en el contexto recuperado; NUNCA cites "
    "articulos o normas de memoria. Si el desarrollo reglamentario no esta en el contexto, "
    "no lo inventes ni lo insinues."
)


def _sistema_consulta_general():
    """Guardrail estricto (cero alucinaciones) para el chat global sin caso."""
    return GUARDRAIL_CONSULTA_GENERAL


def _sistema_chat():
    return (
        "Eres un asistente experto en contrataciones publicas. Respondes consultas de areas "
        "usuarias, especialistas y operadores. Reglas:\n"
        "1. Usa como contexto los DOCUMENTOS DEL CASO (fuentes activas) y las NORMAS RECUPERADAS "
        "del marco legal. No uses conocimiento externo.\n"
        "2. Si la respuesta no esta en el contexto, dilo claramente: no inventes.\n"
        "3. Cita SIEMPRE el respaldo: articulos (ej: Reglamento, Art. 304) y/o la fuente del caso.\n"
        "4. Lenguaje claro y preciso; enumera plazos, montos o pasos cuando aplique."
    )


def _sistema_auditoria(etiquetas):
    etiquetas_txt = ", ".join(etiquetas) if etiquetas else "(sin etiquetas especificas)"
    return (
        "Eres un especialista legal en contrataciones publicas. Analiza el/los documento(s) "
        "adjunto(s) (fuentes activas) teniendo en cuenta que el usuario los ha clasificado con "
        f"las siguientes etiquetas de control: {etiquetas_txt}. Cruza el texto con las normas "
        "recuperadas e identifica riesgos, omisiones o alertas especificas que afecten a los "
        "criterios de esas etiquetas bajo el marco de la Ley 32069. Estructura la respuesta por "
        "etiqueta, cita los articulos de respaldo (ej: Reglamento, Art. 304) y se claro y "
        "accionable. Si algo no puede verificarse con las normas recuperadas, indicalo."
    )


def _ensamblar_prompt(pregunta, filas, modo, contexto_fuentes, etiquetas):
    """Arma (system_instruction, prompt) EXACTAMENTE como el endpoint /api/chat de hoy,
    segun el modo (general / chat / analisis). Pura, SIN red: es la base verificable de
    la no-regresion del endpoint."""
    contexto_normas = construir_contexto(filas) if filas else "(sin normas recuperadas)"
    if modo == "general":
        # EL CANDADO: el guardrail va como system_instruction Y antepuesto al contexto.
        prompt = (
            GUARDRAIL_CONSULTA_GENERAL + "\n\n"
            "CONTEXTO NORMATIVO PROPORCIONADO (unica fuente de verdad):\n"
            + contexto_normas + "\n\n"
            + INSTRUCCION_PRECISION + "\n\n"
            + INSTRUCCION_ESTRUCTURA + "\n\n"
            + INSTRUCCION_CRUCE + "\n\n"
            + INSTRUCCION_CITAS + "\n\n"
            "CONSULTA DEL USUARIO:\n" + pregunta
        )
        return _sistema_consulta_general(), prompt

    # ===== modos de CASO: chat / analisis =====
    secciones = [f"NORMAS RECUPERADAS (base vectorial):\n{contexto_normas}"]
    if contexto_fuentes:
        secciones.append("DOCUMENTOS DEL CASO (fuentes activas seleccionadas por el usuario):\n"
                         + contexto_fuentes)
    else:
        secciones.append("DOCUMENTOS DEL CASO: (ninguna fuente activa en este turno).")

    if modo == "analisis":
        sistema = _sistema_auditoria(etiquetas)
        secciones.append("TAREA: Realiza la auditoria legal de las fuentes activas segun las "
                         "instrucciones del sistema." +
                         (f"\nFoco adicional del usuario: {pregunta}" if pregunta else ""))
    else:
        sistema = _sistema_chat()
        secciones.append(f"CONSULTA DEL USUARIO:\n{pregunta or '(resume y comenta las fuentes activas)'}")

    secciones.append(INSTRUCCION_PRECISION)
    secciones.append(INSTRUCCION_ESTRUCTURA)
    secciones.append(INSTRUCCION_CRUCE)
    secciones.append(INSTRUCCION_CITAS)
    return sistema, "\n\n".join(secciones)


def _construir_contents(prompt, historial):
    """Historial multi-turno con roles nativos (user/model) + el prompt final como turno
    de usuario. Identico a la logica previa de _responder_llm (app.py)."""
    contents = []
    for t in list(historial)[-MAX_TURNOS_HISTORIAL:]:
        rol = "model" if (getattr(t, "rol", "") or "").lower() in ("model", "assistant", "ia", "bot") else "user"
        txt = (getattr(t, "texto", "") or "").strip()
        if txt:
            contents.append(Content(role=rol, parts=[Part(text=txt)]))
    while contents and contents[0].role == "model":
        contents.pop(0)
    contents.append(Content(role="user", parts=[Part(text=prompt)]))
    return contents


def generar_respuesta(pregunta, filas, *, modo="general", contexto_fuentes="",
                      etiquetas=(), historial=(), temperatura=0.2, modelo=MODELO_GEN):
    """Funcion UNICA de generacion: la usan el endpoint /api/chat, el CLI y el eval.
    Arma el prompt + system_instruction EXACTAMENTE como /api/chat en los 3 modos
    (general / chat / analisis), inyecta historial con roles nativos, genera con
    temperatura parametrizable (0.2 = produccion) y pasa la salida por verificar_citas.
    Devuelve {"respuesta", "citas_validas", "citas_invalidas"}."""
    sistema, prompt = _ensamblar_prompt(pregunta, filas, modo, contexto_fuentes, etiquetas)
    contents = _construir_contents(prompt, historial)
    resp = con_reintentos(
        lambda: cliente().models.generate_content(
            model=modelo, contents=contents,
            config=GenerateContentConfig(system_instruction=sistema, temperature=temperatura),
        ),
        etiqueta="generacion")
    chequeo = verificar_citas(resp.text, len(filas))
    return {"respuesta": chequeo["respuesta_limpia"],
            "citas_validas": chequeo["citas_validas"],
            "citas_invalidas": chequeo["citas_invalidas"]}


def responder(pregunta, k=TOP_K, modelo=MODELO_GEN, temperatura=0.2):
    """RAG por CLI: recupera en modo general y genera con la funcion UNIFICADA
    (mismo prompt que el endpoint). `temperatura` 0.2 por defecto (produccion); el eval
    puede fijar 0 sin alterar el default. Devuelve (respuesta_limpia, filas)."""
    filas = recuperar(consultas_busqueda(pregunta), k)
    if not filas:
        return "No se encontraron fragmentos normativos relevantes.", []
    res = generar_respuesta(pregunta, filas, modo="general", temperatura=temperatura, modelo=modelo)
    if res["citas_invalidas"]:
        print(f"[!] Citas inventadas neutralizadas: {res['citas_invalidas']}")
    return res["respuesta"], filas


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
        print(f"  [{i}] {formato_cita(f)}: {f['articulo_titulo']} "
              f"(coseno {1 - f['distance']:.3f})")


if __name__ == "__main__":
    main()
