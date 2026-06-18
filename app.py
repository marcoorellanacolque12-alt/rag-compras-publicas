"""
app.py
-------------------------------------------------------------------
Asistente RAG de Contrataciones Publicas — "Repositorio de Casos"
(experiencia tipo NotebookLM con persistencia local).

Flujo:
  - Repositorio de Casos: el usuario ve sus casos guardados y puede crear
    uno nuevo. Al entrar a un caso, gestiona SUS fuentes y chatea con ellas.
  - Aislamiento: cada caso tiene sus propias fuentes y su propio chat.
    Un caso nunca mezcla documentos con otro (todo se filtra por caso_id).

Persistencia: SQLite nativo (casos.db), dos tablas: casos y fuentes.

Subida de fuentes: asincrona con estado (procesando -> listo/error),
con extraccion de texto y FALLBACK de OCR (Cloud Vision) para escaneos.

Memoria multi-turno: el historial se inyecta con roles nativos user/model.

Endpoints:
  GET    /                                  -> interfaz web.
  GET    /api/casos                         -> lista de casos.
  POST   /api/casos                         -> crea un caso {nombre}.
  DELETE /api/casos/{cid}                   -> elimina un caso (y sus fuentes).
  GET    /api/casos/{cid}/fuentes           -> fuentes del caso.
  POST   /api/casos/{cid}/fuentes           -> sube una fuente (async + OCR).
  GET    /api/casos/{cid}/fuentes/{fid}     -> estado de una fuente (polling).
  DELETE /api/casos/{cid}/fuentes/{fid}     -> elimina una fuente.
  POST   /api/chat                          -> chat/analisis dentro de un caso.
  GET    /salud                             -> healthcheck.

Requisitos:
    pip install fastapi "uvicorn[standard]" python-multipart google-cloud-bigquery \
                google-genai pypdf python-docx pymupdf google-cloud-vision
    Autenticacion ADC + tabla BigQuery + Vision API.

Ejecutar en local:
    python -m uvicorn app:app --host 0.0.0.0 --port 8080
-------------------------------------------------------------------
"""

import io
import os
import glob
import uuid
import json
import sqlite3
import zipfile
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google.genai.types import GenerateContentConfig, Content, Part

# Motor RAG ya implementado y probado.
from responder import (
    recuperar, construir_contexto, cliente, MODELO_GEN,
    GUARDRAIL_CONSULTA_GENERAL, consultas_busqueda,
    embeber_para_indexar, indexar_chunks, eliminar_por_prefijo,
    con_reintentos, doc_label, listar_normas, formato_cita,
    generar_respuesta,   # generacion UNIFICADA (prompt+system identicos en endpoint/CLI/eval)
)
# Motores de extraccion (en memoria): PDF+OCR, DOCX, Excel/CSV->Markdown, imagen->OCR.
from extraccion_texto import (
    extraer_pdf_inteligente, extraer_docx,
    extraer_xlsx, extraer_xls, extraer_csv, ocr_imagen_vision,
)
# Derivacion de fase/emisor para los documentos de la Biblioteca web.
from chunking_articulos import fase_documento, detectar_emisor

@asynccontextmanager
async def lifespan(app):
    # CICLO DE VIDA: pre-crea el cliente GenAI persistente UNA sola vez, en el arranque,
    # para que las BackgroundTasks concurrentes compartan un cliente ya inicializado y
    # nunca disparen la carrera que lo dejaba 'closed'. El cliente vive todo el proceso.
    try:
        cliente()
        print("[lifespan] cliente GenAI persistente pre-inicializado.")
    except Exception as e:
        print(f"[lifespan] aviso: no se pudo pre-crear el cliente GenAI: {e}")
    yield


app = FastAPI(title="Asistente RAG - Repositorio de Casos", lifespan=lifespan)

# ===================== CONFIGURACION =====================
DB_PATH = "casos.db"
BACKUP_DIR = "backups"             # snapshots automaticos de casos.db (gitignored)
MAX_BACKUPS = 10                   # rotacion: se conservan los N mas recientes
_db_lock = threading.Lock()

# Frontend servido desde disco (index.html) + assets estaticos futuros.
STATIC_DIR = "static"
INDEX_HTML_PATH = os.path.join(STATIC_DIR, "index.html")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Formatos aceptados en la subida de fuentes.
EXT_SOPORTADAS = (".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv",
                  ".png", ".jpg", ".jpeg", ".zip")
EXT_INTERNAS_ZIP = tuple(e for e in EXT_SOPORTADAS if e != ".zip")  # dentro de un .zip, sin anidar
ZIP_MAX_INNER_BYTES = 40 * 1024 * 1024  # tope por archivo interno de un ZIP (40 MB)

# Biblioteca Institucional: documentos globales que SI se vectorizan (van al vector store).
EXT_BIBLIOTECA = tuple(e for e in EXT_SOPORTADAS if e != ".zip")     # un .zip no se vectoriza
CATEGORIAS_VALIDAS = {
    "leyes_y_reglamentos", "directivas", "documentos_orientacion",
    "resoluciones_tribunal", "opiniones",
}
CHUNK_MAX_CHARS = 1500     # tamano objetivo de cada fragmento
CHUNK_OVERLAP = 200        # solape entre fragmentos (continuidad de contexto)

# Copia del archivo original SOLO mientras pueda necesitarse para REPROCESAR (se borra al
# indexar con exito). Permite reintentar los fallidos sin volver a subir el lote. Gitignored.
ARCHIVOS_BIBLIOTECA = "biblioteca_archivos"
os.makedirs(ARCHIVOS_BIBLIOTECA, exist_ok=True)

MAX_CONTEXT_CHARS = 200_000        # tope de texto de fuentes activas enviado al LLM
QUERY_DOC_CHARS = 3_000            # extracto para construir la consulta de recuperacion
# La generacion unificada (MAX_TURNOS_HISTORIAL, INSTRUCCION_*, builders _sistema_* y la
# funcion generar_respuesta que arma prompt + system_instruction) vive ahora en responder.py.
# El CLI y el eval convergen hacia el MISMO armado que usa este endpoint.
# ========================================================


# ============================ BASE DE DATOS (SQLite) ============================
def _conn():
    con = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def _ahora():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _init_db():
    con = _conn()
    try:
        con.execute("PRAGMA journal_mode = WAL")
        con.execute("""
            CREATE TABLE IF NOT EXISTS casos (
                id             TEXT PRIMARY KEY,
                nombre         TEXT NOT NULL,
                fecha_creacion TEXT NOT NULL
            )""")
        con.execute("""
            CREATE TABLE IF NOT EXISTS fuentes (
                id             TEXT PRIMARY KEY,
                caso_id        TEXT NOT NULL,
                nombre         TEXT NOT NULL,
                etiquetas      TEXT,                 -- JSON array
                texto          TEXT,
                n_chars        INTEGER DEFAULT 0,
                metodo         TEXT,
                estado         TEXT NOT NULL,         -- procesando | listo | error
                error          TEXT,
                fecha_creacion TEXT NOT NULL,
                FOREIGN KEY (caso_id) REFERENCES casos(id) ON DELETE CASCADE
            )""")
        con.execute("""
            CREATE TABLE IF NOT EXISTS biblioteca (
                id             TEXT PRIMARY KEY,
                nombre         TEXT NOT NULL,
                categoria      TEXT,
                anio           INTEGER,
                vigente        INTEGER DEFAULT 1,     -- 1 vigente | 0 derogada
                n_chunks       INTEGER DEFAULT 0,
                metodo         TEXT,
                estado         TEXT NOT NULL,          -- procesando | listo | error
                error          TEXT,
                fecha_creacion TEXT NOT NULL
            )""")
        con.commit()
    finally:
        con.close()


# --- Casos ---
def crear_caso(nombre):
    cid = uuid.uuid4().hex[:12]
    ts = _ahora()
    with _db_lock:
        con = _conn()
        try:
            con.execute("INSERT INTO casos (id, nombre, fecha_creacion) VALUES (?,?,?)",
                        (cid, nombre, ts))
            con.commit()
        finally:
            con.close()
    return {"id": cid, "nombre": nombre, "fecha_creacion": ts, "n_fuentes": 0}


def listar_casos():
    con = _conn()
    try:
        rows = con.execute("""
            SELECT c.id, c.nombre, c.fecha_creacion, COUNT(f.id) AS n_fuentes
            FROM casos c LEFT JOIN fuentes f ON f.caso_id = c.id
            GROUP BY c.id ORDER BY c.fecha_creacion DESC
        """).fetchall()
    finally:
        con.close()
    return [dict(r) for r in rows]


def caso_existe(cid):
    con = _conn()
    try:
        return con.execute("SELECT 1 FROM casos WHERE id=?", (cid,)).fetchone() is not None
    finally:
        con.close()


def borrar_caso(cid):
    with _db_lock:
        con = _conn()
        try:
            con.execute("DELETE FROM fuentes WHERE caso_id=?", (cid,))  # explicito (ademas del cascade)
            con.execute("DELETE FROM casos WHERE id=?", (cid,))
            con.commit()
        finally:
            con.close()


def borrar_casos_masivo(ids):
    """Borrado MASIVO: DELETE ... WHERE id IN (...). Devuelve cuantos casos se eliminaron."""
    ids = [i for i in (ids or []) if i]
    if not ids:
        return 0
    marcas = ",".join("?" * len(ids))                   # placeholders parametrizados (no input crudo en SQL)
    with _db_lock:
        con = _conn()
        try:
            con.execute(f"DELETE FROM fuentes WHERE caso_id IN ({marcas})", ids)  # explicito + ON DELETE CASCADE
            cur = con.execute(f"DELETE FROM casos WHERE id IN ({marcas})", ids)
            con.commit()
            return cur.rowcount
        finally:
            con.close()


# --- Fuentes ---
def crear_fuente_registro(caso_id, nombre, etiquetas_list):
    fid = uuid.uuid4().hex[:12]
    ts = _ahora()
    with _db_lock:
        con = _conn()
        try:
            con.execute("""INSERT INTO fuentes
                (id, caso_id, nombre, etiquetas, texto, n_chars, metodo, estado, error, fecha_creacion)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (fid, caso_id, nombre, json.dumps(etiquetas_list, ensure_ascii=False),
                 "", 0, None, "procesando", None, ts))
            con.commit()
        finally:
            con.close()
    return fid


def actualizar_fuente(fid, **campos):
    if not campos:
        return
    cols = ", ".join(f"{k}=?" for k in campos)          # claves internas, no input de usuario
    vals = list(campos.values()) + [fid]
    with _db_lock:
        con = _conn()
        try:
            con.execute(f"UPDATE fuentes SET {cols} WHERE id=?", vals)
            con.commit()
        finally:
            con.close()


def _fila_publica(r):
    return {
        "id": r["id"], "caso_id": r["caso_id"], "nombre": r["nombre"],
        "etiquetas": json.loads(r["etiquetas"] or "[]"),
        "n_chars": r["n_chars"], "metodo": r["metodo"],
        "estado": r["estado"], "error": r["error"],
    }


def obtener_fuente_publica(fid):
    con = _conn()
    try:
        r = con.execute("SELECT * FROM fuentes WHERE id=?", (fid,)).fetchone()
    finally:
        con.close()
    return _fila_publica(r) if r else None


def listar_fuentes_publicas(caso_id):
    con = _conn()
    try:
        rows = con.execute(
            "SELECT * FROM fuentes WHERE caso_id=? ORDER BY fecha_creacion ASC", (caso_id,)).fetchall()
    finally:
        con.close()
    return [_fila_publica(r) for r in rows]


def fuentes_activas_texto(caso_id, ids):
    """Devuelve (con texto) solo las fuentes del caso, en 'ids', y en estado 'listo'."""
    ids = [i for i in (ids or []) if i]
    if not ids:
        return []
    marcas = ",".join("?" * len(ids))
    con = _conn()
    try:
        rows = con.execute(
            f"""SELECT * FROM fuentes
                WHERE caso_id=? AND estado='listo' AND id IN ({marcas})""",
            [caso_id] + ids).fetchall()
    finally:
        con.close()
    return [{"id": r["id"], "nombre": r["nombre"],
             "etiquetas": json.loads(r["etiquetas"] or "[]"), "texto": r["texto"]} for r in rows]


def borrar_fuente(caso_id, fid):
    with _db_lock:
        con = _conn()
        try:
            con.execute("DELETE FROM fuentes WHERE id=? AND caso_id=?", (fid, caso_id))
            con.commit()
        finally:
            con.close()


# --- Biblioteca Institucional (documentos globales vectorizados) ---
def _ruta_archivo_biblio(bid):
    """Ruta de la copia del archivo original (para reprocesar fallidos)."""
    return os.path.join(ARCHIVOS_BIBLIOTECA, bid)


def crear_biblio_registro(nombre, categoria, anio, vigente):
    bid = uuid.uuid4().hex[:12]
    ts = _ahora()
    with _db_lock:
        con = _conn()
        try:
            con.execute("""INSERT INTO biblioteca
                (id, nombre, categoria, anio, vigente, n_chunks, metodo, estado, error, fecha_creacion)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (bid, nombre, categoria, anio, 1 if vigente else 0, 0, None, "procesando", None, ts))
            con.commit()
        finally:
            con.close()
    return bid


def actualizar_biblio(bid, **campos):
    if not campos:
        return
    cols = ", ".join(f"{k}=?" for k in campos)          # claves internas, no input de usuario
    vals = list(campos.values()) + [bid]
    with _db_lock:
        con = _conn()
        try:
            con.execute(f"UPDATE biblioteca SET {cols} WHERE id=?", vals)
            con.commit()
        finally:
            con.close()


def _biblio_publica(r):
    return {
        "id": r["id"], "nombre": r["nombre"], "categoria": r["categoria"],
        "anio": r["anio"], "vigente": bool(r["vigente"]), "n_chunks": r["n_chunks"],
        "metodo": r["metodo"], "estado": r["estado"], "error": r["error"],
    }


def listar_biblio():
    con = _conn()
    try:
        rows = con.execute("SELECT * FROM biblioteca ORDER BY fecha_creacion DESC").fetchall()
    finally:
        con.close()
    return [_biblio_publica(r) for r in rows]


def obtener_biblio(bid):
    con = _conn()
    try:
        r = con.execute("SELECT * FROM biblioteca WHERE id=?", (bid,)).fetchone()
    finally:
        con.close()
    return _biblio_publica(r) if r else None


def borrar_biblio(bid):
    with _db_lock:
        con = _conn()
        try:
            con.execute("DELETE FROM biblioteca WHERE id=?", (bid,))
            con.commit()
        finally:
            con.close()


def _checkpoint_wal():
    """Pliega el WAL dentro de casos.db y lo trunca: deja la base consolidada en disco."""
    con = _conn()
    try:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.commit()
    finally:
        con.close()


def _backup_db():
    """Snapshot CONSISTENTE de casos.db con la API de backup online de SQLite (segura
    aun con WAL activo) y rotacion. No respalda bases vacias (para no desplazar backups
    utiles con instantaneas vacias). Devuelve la ruta creada o None."""
    if not os.path.exists(DB_PATH):
        return None
    con = _conn()
    try:
        n = con.execute("SELECT count(*) FROM casos").fetchone()[0]
    finally:
        con.close()
    if not n:
        return None

    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    destino = os.path.join(BACKUP_DIR, f"casos-{ts}.db")
    src = sqlite3.connect(DB_PATH, timeout=30)
    dst = sqlite3.connect(destino)
    try:
        with dst:
            src.backup(dst)            # copia atomica y consistente (incluye el WAL pendiente)
    finally:
        src.close()
        dst.close()

    # Rotacion: conserva solo los MAX_BACKUPS mas recientes (orden lexicografico = cronologico).
    backups = sorted(glob.glob(os.path.join(BACKUP_DIR, "casos-*.db")))
    for viejo in backups[:-MAX_BACKUPS]:
        try:
            os.remove(viejo)
        except OSError:
            pass
    return destino


def _arranque_db():
    """Arranque a prueba de fallos: crea tablas, respalda y consolida la base.
    Un fallo de backup/checkpoint NUNCA impide que el servidor levante."""
    _init_db()
    try:
        ruta = _backup_db()
        print(f"[arranque] backup creado: {ruta}" if ruta
              else "[arranque] sin backup (base vacia o inexistente).")
    except Exception as e:
        print(f"[arranque] backup omitido: {type(e).__name__}: {e}")
    try:
        _checkpoint_wal()
        print("[arranque] checkpoint WAL (TRUNCATE) aplicado.")
    except Exception as e:
        print(f"[arranque] checkpoint WAL omitido: {type(e).__name__}: {e}")


_arranque_db()  # inicializa tablas + backup automatico + checkpoint WAL al cargar el modulo


# ============================ MODELOS ============================
class NuevoCaso(BaseModel):
    nombre: str = ""


class BorradoMasivo(BaseModel):
    ids: list[str] = []


class Turno(BaseModel):
    rol: str = "user"
    texto: str = ""


class Filtros(BaseModel):
    categorias: list[str] = []          # valores de la columna `categoria` (prefijos normativos)
    excluir_derogada: bool = True       # solo normativa vigente
    anio: str = "Todos"                 # Todos | 2026 | 2025 | 2024 | Anteriores
    normas: list[str] | None = None     # doc_ids seleccionados (None = todas; [] = ninguna)


class Mensaje(BaseModel):
    caso_id: str = ""
    pregunta: str = ""
    fuentes_activas: list[str] = []
    modo: str = "chat"
    k: int = 10                    # fragmentos a recuperar (configurable; antes 5)
    historial: list[Turno] = []
    general: bool = False          # True = Consulta General (chat global sin caso, guardrail estricto)
    filtros: Filtros | None = None  # Busqueda hibrida: pre-filtering por metadatos del vector


# ============================ HELPERS RAG ============================
def _fuentes_normativas(filas):
    # 'numero' = orden 1-based, coincide con [Fragmento N] de construir_contexto y con
    # los marcadores [N] que el modelo coloca en la respuesta.
    return [{
        "numero": i,
        "documento": doc_label(f["documento"], f.get("chunk_id")),   # etiquetado unificado
        "cita": formato_cita(f),                                      # cita natural lista para mostrar
        "tipo_referencia": f.get("tipo_referencia"),
        "referencia": f.get("referencia"),
        "fase": f.get("fase"),
        "emisor": f.get("emisor"),
        "texto": f["texto"],                                          # texto TEXTUAL del fragmento
        "articulo_num": f["articulo_num"],
        "articulo_titulo": f["articulo_titulo"],
        "relevancia": round(1 - f["distance"], 3),
    } for i, f in enumerate(filas, start=1)]


def _extraer_texto(nombre: str, data: bytes):
    """Dispatcher de extraccion EN MEMORIA segun extension. Devuelve (texto, metodo).
       metodo in {nativo, ocr, tabla}. No maneja .zip (ver _procesar_zip)."""
    lower = (nombre or "").lower()
    # A) Documentos de texto.
    if lower.endswith(".pdf"):
        texto, _, metodo = extraer_pdf_inteligente(data)   # nativo con fallback OCR
        return texto, metodo
    if lower.endswith(".docx"):
        texto, _ = extraer_docx(data)                      # parrafos + tablas
        return texto, "nativo"
    if lower.endswith(".doc"):
        # python-docx NO abre el formato binario .doc (OLE, Word 97-2003).
        try:
            texto, _ = extraer_docx(data)
            return texto, "nativo"
        except Exception:
            raise ValueError("El formato .doc binario (Word 97-2003) no es legible directamente. "
                             "Conviertelo a .docx o PDF y vuelve a subirlo.")
    # B) Hojas de calculo / datos tabulares -> Tabla Markdown.
    if lower.endswith(".xlsx"):
        texto, _ = extraer_xlsx(data); return texto, "tabla"
    if lower.endswith(".xls"):
        texto, _ = extraer_xls(data); return texto, "tabla"
    if lower.endswith(".csv"):
        texto, _ = extraer_csv(data); return texto, "tabla"
    # C) Imagenes de evidencias / catalogos -> OCR Cloud Vision.
    if lower.endswith((".png", ".jpg", ".jpeg")):
        return ocr_imagen_vision(data), "ocr"
    raise ValueError("Formato no soportado.")


def _procesar_fuente(fid, caso_id, nombre, data, etiquetas):
    """Worker en segundo plano: extrae texto segun tipo (con OCR/tablas) y actualiza
       el estado en la BD. Los .zip se expanden en fuentes independientes."""
    try:
        if (nombre or "").lower().endswith(".zip"):
            _procesar_zip(fid, caso_id, nombre, data, etiquetas)
            return
        texto, metodo = _extraer_texto(nombre, data)
        if not texto or not texto.strip():
            actualizar_fuente(fid, estado="error",
                              error="No se pudo extraer texto util del documento (vacio o ilegible).")
            return
        actualizar_fuente(fid, texto=texto, n_chars=len(texto), metodo=metodo, estado="listo", error=None)
    except ValueError as e:
        actualizar_fuente(fid, estado="error", error=str(e))
    except Exception as e:
        actualizar_fuente(fid, estado="error", error=f"{type(e).__name__}: {e}")


def _procesar_zip(fid, caso_id, nombre_zip, data, etiquetas):
    """Descomprime un .zip EN MEMORIA y crea una fuente INDEPENDIENTE por cada archivo
       interno soportado (mismo caso_id). La fuente del .zip queda como contenedor con
       un resumen. No recursa zips anidados (proteccion anti zip-bomb)."""
    base_zip = os.path.basename(nombre_zip) or "paquete.zip"
    procesados, errores, ignorados = [], [], []
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        actualizar_fuente(fid, estado="error", metodo="zip",
                          error="El archivo .zip esta danado o no es un ZIP valido.")
        return

    with zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            base = os.path.basename(info.filename)
            if not base or base.startswith("."):
                continue
            lower = base.lower()
            if lower.endswith(".zip"):
                ignorados.append(base + " (zip anidado)"); continue
            if not lower.endswith(EXT_INTERNAS_ZIP):
                ignorados.append(base); continue
            if info.file_size > ZIP_MAX_INNER_BYTES:
                ignorados.append(base + " (>40MB)"); continue
            try:
                inner = zf.read(info)
            except Exception as e:
                errores.append(f"{base}: lectura ({e})"); continue

            # Cada archivo interno = una fuente independiente del caso.
            hijo_fid = crear_fuente_registro(caso_id, base, list(etiquetas) + [f"zip:{base_zip}"])
            try:
                texto, metodo = _extraer_texto(base, inner)
                if not texto or not texto.strip():
                    actualizar_fuente(hijo_fid, estado="error", error="Sin texto extraible.")
                    errores.append(base + " (sin texto)")
                else:
                    actualizar_fuente(hijo_fid, texto=texto, n_chars=len(texto),
                                      metodo=metodo, estado="listo", error=None)
                    procesados.append(base)
            except ValueError as e:
                actualizar_fuente(hijo_fid, estado="error", error=str(e)); errores.append(f"{base}: {e}")
            except Exception as e:
                actualizar_fuente(hijo_fid, estado="error", error=f"{type(e).__name__}: {e}")
                errores.append(f"{base}: {type(e).__name__}")

    if not procesados and not errores:
        actualizar_fuente(fid, estado="error", metodo="zip",
                          error="El ZIP no contenia archivos soportados.")
        return
    resumen = (f"Paquete ZIP '{base_zip}': {len(procesados)} archivo(s) extraido(s) como "
               f"fuentes independientes.\n"
               f"- Procesados: {', '.join(procesados) or '-'}\n"
               f"- Con error: {', '.join(errores) or '-'}\n"
               f"- Ignorados: {', '.join(ignorados) or '-'}")
    actualizar_fuente(fid, texto=resumen, n_chars=len(resumen), metodo="zip",
                      estado="listo", error=None)


def _chunk_texto(texto, max_chars=CHUNK_MAX_CHARS, overlap=CHUNK_OVERLAP):
    """Fragmentacion GENERICA por longitud con solape, cortando en limite de palabra/linea."""
    texto = (texto or "").strip()
    if not texto:
        return []
    trozos, i, n = [], 0, len(texto)
    while i < n:
        fin = min(i + max_chars, n)
        if fin < n:  # intenta cortar en un salto de linea o espacio cercano (no a mitad de palabra)
            corte = texto.rfind("\n", i, fin)
            if corte <= i + max_chars // 2:
                corte = texto.rfind(" ", i, fin)
            if corte > i:
                fin = corte
        trozo = texto[i:fin].strip()
        if trozo:
            trozos.append(trozo)
        if fin >= n:
            break
        i = max(fin - overlap, i + 1)
    return trozos


def _procesar_biblioteca(bid, nombre, data, categoria, anio, vigente):
    """Worker: extrae -> fragmenta -> embebe -> INSERTA en el vector store (con metadatos)."""
    try:
        texto, metodo = _extraer_texto(nombre, data)
        if not texto or not texto.strip():
            actualizar_biblio(bid, estado="error", error="Sin texto extraible del documento.")
            return
        trozos = _chunk_texto(texto)
        if not trozos:
            actualizar_biblio(bid, estado="error", error="No se generaron fragmentos.")
            return
        vectores = embeber_para_indexar(trozos)
        # Metadatos por documento (la Biblioteca web usa troceo generico por tamaño:
        # tipo_referencia="seccion"/sin referencia; fase y emisor se derivan).
        fase = fase_documento(categoria, nombre)
        emisor = detectar_emisor(nombre, texto, categoria)
        filas = [{
            "chunk_id": f"bib_{bid}__c{idx:04d}",
            "categoria": categoria,
            "documento": nombre,
            "tipo_referencia": "seccion",
            "referencia": None,
            "articulo_num": None,
            "articulo_titulo": f"{nombre} (parte {idx + 1})",
            "parte": idx,
            "n_chars": len(t),
            "texto": t,
            "fase": fase,
            "emisor": emisor,
            "anio": anio,
            "vigente": bool(vigente),
            "embedding": v,
        } for idx, (t, v) in enumerate(zip(trozos, vectores))]
        indexar_chunks(filas)
        actualizar_biblio(bid, n_chunks=len(filas), metodo=metodo, estado="listo", error=None)
        # Exito: ya no hace falta la copia para reproceso.
        try:
            os.remove(_ruta_archivo_biblio(bid))
        except OSError:
            pass
    except Exception as e:
        # Registro silencioso: se anota el fallo (consola + estado) pero NO se propaga;
        # los demas archivos de la cola (otras BackgroundTasks) siguen su curso.
        print(f"[biblioteca] FALLO al indexar '{nombre}': {type(e).__name__}: {e}")
        actualizar_biblio(bid, estado="error", error=f"{type(e).__name__}: {e}")


def _historial_texto(historial, n=4):
    """Ultimos n turnos como texto plano, para REFORMULAR la consulta (resolver
    follow-ups). Es la misma conversacion en curso; no requiere base de datos."""
    out = []
    for t in (historial or [])[-n:]:
        rol = "Asistente" if (t.rol or "").lower() in ("model", "assistant", "ia", "bot") else "Usuario"
        txt = " ".join((t.texto or "").split())[:400]
        if txt:
            out.append(f"{rol}: {txt}")
    return "\n".join(out)


def _mensaje_error_llm(e):
    """Traduce un fallo de generacion en un mensaje claro para el usuario (sin 500 generico)."""
    msg = str(e).lower()
    if ("se agotaron los reintentos" in msg or "resource_exhausted" in msg
            or "429" in msg or "quota" in msg or "rate limit" in msg):
        return ("El servicio de IA está temporalmente saturado (límite de tasa). "
                "Espera unos segundos y vuelve a intentar.")
    if any(s in msg for s in ("safety", "blocked", "block_reason", "finish_reason",
                              "prohibited", "recitation")):
        return ("La respuesta fue bloqueada por los filtros de seguridad del modelo. "
                "Reformula la consulta.")
    return f"No se pudo generar la respuesta ({type(e).__name__}). Inténtalo nuevamente."


# ============================ ENDPOINTS ============================
@app.get("/salud")
def salud():
    return {"status": "ok", "casos": len(listar_casos())}


# ---- Casos ----
@app.get("/api/casos")
def api_listar_casos():
    return listar_casos()


@app.post("/api/casos")
def api_crear_caso(c: NuevoCaso):
    nombre = (c.nombre or "").strip() or "Caso sin nombre"
    return JSONResponse(status_code=201, content=crear_caso(nombre))


# IMPORTANTE: la ruta literal /bulk se declara ANTES de /{cid} para que no
# sea capturada por el path param (cid="bulk").
@app.delete("/api/casos/bulk")
def api_borrar_casos_bulk(b: BorradoMasivo):
    eliminados = borrar_casos_masivo(b.ids)
    return {"ok": True, "eliminados": eliminados}


@app.delete("/api/casos/{cid}")
def api_borrar_caso(cid: str):
    borrar_caso(cid)
    return {"ok": True}


# ---- Fuentes (dentro de un caso) ----
@app.get("/api/casos/{cid}/fuentes")
def api_listar_fuentes(cid: str):
    if not caso_existe(cid):
        return JSONResponse(status_code=404, content={"error": "Caso no encontrado."})
    return listar_fuentes_publicas(cid)


@app.post("/api/casos/{cid}/fuentes")
async def api_subir_fuente(cid: str, background_tasks: BackgroundTasks,
                           archivo: UploadFile = File(...), etiquetas: str = Form("")):
    try:
        if not caso_existe(cid):
            return JSONResponse(status_code=404, content={"error": "Caso no encontrado."})

        lista_etiquetas = [e.strip() for e in (etiquetas or "").split(",") if e.strip()]

        lower = (archivo.filename or "").lower()
        if not lower.endswith(EXT_SOPORTADAS):
            return JSONResponse(status_code=400, content={
                "error": "Formato no soportado. Admitidos: pdf, docx, doc, xlsx, xls, csv, "
                         "png, jpg, jpeg, zip."})

        data = await archivo.read()
        if not data:
            return JSONResponse(status_code=400, content={"error": "El archivo llego vacio."})

        fid = crear_fuente_registro(cid, archivo.filename, lista_etiquetas)
        background_tasks.add_task(_procesar_fuente, fid, cid, archivo.filename, data, lista_etiquetas)
        return JSONResponse(status_code=202, content=obtener_fuente_publica(fid))

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": f"Error interno al registrar el documento: {type(e).__name__}: {e}"})


@app.get("/api/casos/{cid}/fuentes/{fid}")
def api_estado_fuente(cid: str, fid: str):
    pub = obtener_fuente_publica(fid)
    if pub is None or pub["caso_id"] != cid:
        return JSONResponse(status_code=404, content={"error": "Fuente no encontrada en este caso."})
    return pub


@app.delete("/api/casos/{cid}/fuentes/{fid}")
def api_borrar_fuente(cid: str, fid: str):
    borrar_fuente(cid, fid)
    return {"ok": True}


# ---- Biblioteca Institucional (documentos globales -> vector store con metadatos) ----
@app.get("/api/biblioteca")
def api_listar_biblioteca():
    return listar_biblio()


@app.post("/api/biblioteca")
async def api_subir_biblioteca(background_tasks: BackgroundTasks,
                               archivo: UploadFile = File(...),
                               categoria: str = Form(""),
                               anio: int = Form(0),
                               vigente: bool = Form(True)):
    try:
        lower = (archivo.filename or "").lower()
        if not lower.endswith(EXT_BIBLIOTECA):
            return JSONResponse(status_code=400, content={
                "error": "Formato no soportado. Admitidos: pdf, docx, doc, xlsx, xls, csv, "
                         "png, jpg, jpeg (el .zip no se vectoriza; sube los archivos individuales)."})
        if categoria not in CATEGORIAS_VALIDAS:
            return JSONResponse(status_code=400, content={
                "error": "Selecciona una Categoría normativa válida."})

        data = await archivo.read()
        if not data:
            return JSONResponse(status_code=400, content={"error": "El archivo llego vacio."})

        anio_val = anio if anio and anio > 0 else None
        bid = crear_biblio_registro(archivo.filename, categoria, anio_val, vigente)
        # Copia para poder REPROCESAR si la indexacion falla (se borra al tener exito).
        try:
            with open(_ruta_archivo_biblio(bid), "wb") as fh:
                fh.write(data)
        except OSError as e:
            print(f"[biblioteca] no se pudo guardar copia de '{archivo.filename}': {e}")
        background_tasks.add_task(_procesar_biblioteca, bid, archivo.filename, data,
                                  categoria, anio_val, vigente)
        return JSONResponse(status_code=202, content=obtener_biblio(bid))
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": f"Error interno al registrar el documento: {type(e).__name__}: {e}"})


def _reencolar_biblio(doc, background_tasks):
    """Reprograma la indexacion de un documento en error usando su copia guardada.
    Devuelve True si se reencolo; False si no hay copia para reprocesar."""
    ruta = _ruta_archivo_biblio(doc["id"])
    if not os.path.exists(ruta):
        return False
    with open(ruta, "rb") as fh:
        data = fh.read()
    actualizar_biblio(doc["id"], estado="procesando", error=None)
    background_tasks.add_task(_procesar_biblioteca, doc["id"], doc["nombre"], data,
                              doc["categoria"], doc["anio"], bool(doc["vigente"]))
    return True


# Rutas literales declaradas ANTES de /{bid} para que no las capture el path param.
@app.post("/api/biblioteca/reintentar-fallidos")
def api_reintentar_fallidos(background_tasks: BackgroundTasks):
    """Reintenta SOLO los documentos en estado 'error' (sin re-subir el lote)."""
    reintentados, sin_copia = 0, 0
    for d in listar_biblio():
        if d["estado"] != "error":
            continue
        if _reencolar_biblio(d, background_tasks):
            reintentados += 1
        else:
            sin_copia += 1
    return {"ok": True, "reintentados": reintentados, "sin_copia": sin_copia}


@app.post("/api/biblioteca/{bid}/reintentar")
def api_reintentar_biblioteca(bid: str, background_tasks: BackgroundTasks):
    doc = obtener_biblio(bid)
    if doc is None:
        return JSONResponse(status_code=404, content={"error": "Documento no encontrado."})
    if doc["estado"] != "error":
        return JSONResponse(status_code=400, content={
            "error": "Solo se reintentan documentos en estado 'error'."})
    if not _reencolar_biblio(doc, background_tasks):
        return JSONResponse(status_code=400, content={
            "error": "No hay copia del archivo para reprocesar; vuelve a subirlo."})
    return obtener_biblio(bid)


@app.get("/api/biblioteca/{bid}")
def api_estado_biblioteca(bid: str):
    pub = obtener_biblio(bid)
    if pub is None:
        return JSONResponse(status_code=404, content={"error": "Documento no encontrado."})
    return pub


@app.delete("/api/biblioteca/{bid}")
def api_borrar_biblioteca(bid: str):
    # Borra primero los vectores del documento en BigQuery, luego el registro local.
    try:
        eliminar_por_prefijo(f"bib_{bid}__")
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": f"No se pudieron borrar los vectores: {type(e).__name__}: {e}"})
    borrar_biblio(bid)
    try:
        os.remove(_ruta_archivo_biblio(bid))   # elimina la copia si existia
    except OSError:
        pass
    return {"ok": True}


# ---- Normas (para el menu de seleccion individual) ----
@app.get("/api/normas")
def api_listar_normas():
    try:
        return listar_normas()
    except Exception as e:
        return JSONResponse(status_code=503, content={
            "error": f"No se pudieron listar las normas: {type(e).__name__}: {e}"})


# ---- Chat / Analisis (aislado por caso) ----
@app.post("/api/chat")
def chat(m: Mensaje):
    pregunta = (m.pregunta or "").strip()
    filtros = m.filtros.model_dump() if m.filtros else None   # pre-filtering hibrido

    # ===== CONSULTA GENERAL: chat global sin caso, con GUARDRAIL ESTRICTO =====
    if m.general:
        if not pregunta:
            return JSONResponse(status_code=400, content={
                "error": "Escribe una consulta para la Consulta General."})

        hist_txt = _historial_texto(m.historial)
        filas = recuperar(consultas_busqueda(pregunta, hist_txt), k=m.k, filtros=filtros)
        try:
            res = generar_respuesta(pregunta, filas, modo="general", historial=m.historial)
        except Exception as e:
            return JSONResponse(status_code=503, content={"error": _mensaje_error_llm(e)})
        return {
            "respuesta": res["respuesta"],
            "modo": "general",
            "fuentes_usadas": [],
            "fuentes_normativas": _fuentes_normativas(filas),
            "citas_invalidas": res["citas_invalidas"],
        }

    # ===== CHAT / ANALISIS dentro de un caso =====
    if not m.caso_id or not caso_existe(m.caso_id):
        return JSONResponse(status_code=400, content={"error": "Caso no valido o no seleccionado."})

    # AISLAMIENTO: solo fuentes de ESTE caso, encendidas y en estado 'listo'.
    activos = fuentes_activas_texto(m.caso_id, m.fuentes_activas)
    etiquetas = sorted({e for f in activos for e in f["etiquetas"]})

    if not pregunta and not activos:
        return JSONResponse(status_code=400, content={
            "error": "Escribe una consulta o activa al menos una fuente del caso."})

    # Contexto de las fuentes activas (concatenado y acotado).
    partes = []
    for f in activos:
        cab = f"[FUENTE: {f['nombre']} | etiquetas: {', '.join(f['etiquetas']) or '-'}]"
        partes.append(cab + "\n" + f["texto"])
    contexto_fuentes = "\n\n".join(partes)[:MAX_CONTEXT_CHARS]

    # Recuperacion en la base vectorial (guiada por pregunta + etiquetas + extracto).
    # Se expande la consulta (tema/sinonimos + follow-ups) conservando el extracto de fuentes.
    base_query = pregunta or (", ".join(etiquetas))
    extracto = contexto_fuentes[:QUERY_DOC_CHARS]
    qs = consultas_busqueda(base_query, _historial_texto(m.historial))
    consultas = [f"{q}\n{extracto}".strip() for q in qs] or ([extracto] if extracto else [])
    filas = recuperar(consultas, k=m.k, filtros=filtros) if consultas else []

    # Armado + generacion UNIFICADOS (mismo prompt/system que antes vivia inline aqui).
    modo = "analisis" if m.modo == "analisis" else "chat"
    try:
        res = generar_respuesta(pregunta, filas, modo=modo, contexto_fuentes=contexto_fuentes,
                                etiquetas=etiquetas, historial=m.historial)
    except Exception as e:
        return JSONResponse(status_code=503, content={"error": _mensaje_error_llm(e)})

    return {
        "respuesta": res["respuesta"],
        "modo": m.modo,
        "fuentes_usadas": [{"id": f["id"], "nombre": f["nombre"]} for f in activos],
        "fuentes_normativas": _fuentes_normativas(filas),
        "citas_invalidas": res["citas_invalidas"],
    }


# ============================== UI =============================
@app.get("/", response_class=HTMLResponse)
def home():
    # Sirve el frontend desde disco (static/index.html). Fallback al HTML embebido
    # para que la raiz NUNCA devuelva 404 aunque el archivo no se haya materializado.
    if os.path.exists(INDEX_HTML_PATH):
        return FileResponse(INDEX_HTML_PATH, media_type="text/html")
    return HTMLResponse(HTML)


HTML = r"""
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Asistente de Contrataciones Públicas — Repositorio de Casos</title>
<!-- Tema (claro/oscuro/sistema) ANTES de pintar, para evitar parpadeo -->
<script>(function(){try{var m=localStorage.getItem('tema')||'sistema';
  document.documentElement.setAttribute('data-theme',{claro:'light',oscuro:'dark',sistema:'system'}[m]||'system');
}catch(e){document.documentElement.setAttribute('data-theme','system');}})();</script>
<script src="https://cdn.tailwindcss.com"></script>
<script>
  // RE-SKIN "Grafito y cobre": remapea la paleta de Tailwind a variables CSS (tokens).
  // Asi todo el markup existente (slate/blue/amber/emerald) toma los colores del tema.
  tailwind.config = { theme: { extend: { colors: {
    slate: {
      50:'rgb(var(--c100)/<alpha-value>)',  100:'rgb(var(--c100)/<alpha-value>)',
      200:'rgb(var(--c200)/<alpha-value>)', 300:'rgb(var(--c300)/<alpha-value>)',
      400:'rgb(var(--c400)/<alpha-value>)', 500:'rgb(var(--c500)/<alpha-value>)',
      600:'rgb(var(--c600)/<alpha-value>)', 700:'rgb(var(--c700)/<alpha-value>)',
      800:'rgb(var(--c800)/<alpha-value>)', 900:'rgb(var(--c900)/<alpha-value>)',
      950:'rgb(var(--c950)/<alpha-value>)',
    },
    blue:    { 400:'rgb(var(--accent)/<alpha-value>)', 500:'rgb(var(--accent2)/<alpha-value>)', 600:'rgb(var(--accent)/<alpha-value>)' },
    amber:   { 300:'rgb(var(--accent)/<alpha-value>)', 400:'rgb(var(--accent)/<alpha-value>)', 500:'rgb(var(--accent)/<alpha-value>)', 600:'rgb(var(--accent)/<alpha-value>)' },
    emerald: { 300:'rgb(var(--c300)/<alpha-value>)', 400:'rgb(var(--c500)/<alpha-value>)', 500:'rgb(var(--c700)/<alpha-value>)' },
    cobre:   'rgb(var(--accent)/<alpha-value>)',
  } } } };
</script>
<style>
  /* ====== TOKENS "Grafito y cobre" (canales RGB para soportar opacidades) ====== */
  :root {                          /* CLARO (derivado; cobre exacto del spec) */
    --c950:244 243 241; --c900:255 255 255; --c800:236 235 232; --c700:222 219 213;
    --c600:200 197 190; --c500:122 122 130; --c400:106 106 115; --c300:90 90 98;
    --c200:45 45 50;    --c100:38 38 42;
    --accent:180 111 69;  --accent2:156 92 42;   /* #B46F45 / hover */
    color-scheme: light;
  }
  [data-theme="dark"] {            /* OSCURO (valores EXACTOS del spec) */
    --c950:37 40 42;    --c900:45 49 51;   --c800:54 59 62;   --c700:69 75 80;
    --c600:90 96 102;   --c500:138 144 153;--c400:162 162 171;--c300:194 196 201;
    --c200:233 231 227; --c100:240 238 234;
    --accent:197 123 77;  --accent2:210 145 95;   /* #C57B4D / hover */
    color-scheme: dark;
  }
  @media (prefers-color-scheme: dark) {           /* SISTEMA = sigue al SO */
    [data-theme="system"] {
      --c950:37 40 42;    --c900:45 49 51;   --c800:54 59 62;   --c700:69 75 80;
      --c600:90 96 102;   --c500:138 144 153;--c400:162 162 171;--c300:194 196 201;
      --c200:233 231 227; --c100:240 238 234;
      --accent:197 123 77;  --accent2:210 145 95;
      color-scheme: dark;
    }
  }
  body { transition: background-color .15s ease, color .15s ease; }
  .scroll-y { overflow-y: auto; }
  .switch { position: relative; display: inline-block; width: 38px; height: 22px; flex: none; }
  .switch input { opacity: 0; width: 0; height: 0; }
  .slider { position: absolute; cursor: pointer; inset: 0; background: rgb(var(--c600)); border-radius: 9999px; transition: .2s; }
  .slider:before { content: ""; position: absolute; height: 16px; width: 16px; left: 3px; top: 3px; background: #fff; border-radius: 9999px; transition: .2s; }
  input:checked + .slider { background: rgb(var(--accent)); }
  input:checked + .slider:before { transform: translateX(16px); }
  .prosa { white-space: pre-wrap; line-height: 1.6; }
  .hidden-x { display: none; }
  .panel-oculto { display: none !important; }   /* colapso de paneles laterales (responsive-safe) */
  /* Selector de tema (segmented) */
  .tema-btn { cursor: pointer; padding: 4px 7px; border-radius: 6px; line-height: 1; }
  .tema-btn[aria-pressed="true"] { background: rgb(var(--accent) / 0.18); }
  /* Citas estilo NotebookLM: marcadores cobre + panel desplegable */
  .cita-badge { display:inline-flex; align-items:center; justify-content:center; min-width:1.15rem;
    height:1.15rem; padding:0 .3rem; margin:0 .12rem; font-size:.66rem; font-weight:700; line-height:1;
    border-radius:.35rem; cursor:pointer; vertical-align:.12em; background:rgb(var(--accent) / 0.16);
    color:rgb(var(--accent)); border:1px solid rgb(var(--accent) / 0.45); transition:background .15s; }
  .cita-badge:hover { background:rgb(var(--accent) / 0.30); }
  .cita-panel { position:fixed; right:1rem; bottom:1rem; width:min(420px, calc(100vw - 2rem));
    max-height:62vh; overflow-y:auto; background:rgb(var(--c900)); border:1px solid rgb(var(--c700));
    border-radius:.6rem; box-shadow:0 10px 30px rgb(0 0 0 / .30); z-index:50; }
  .cita-quote { border-left:3px solid rgb(var(--accent)); padding-left:.7rem; white-space:pre-wrap;
    line-height:1.55; color:rgb(var(--c200)); }
  .cita-flash { outline:2px solid rgb(var(--accent)); border-radius:.3rem; transition:outline .3s; }
  /* Mide comoda para el chat: limita el ancho de cada turno y lo centra */
  #chat > div { max-width: 56rem; margin-left: auto; margin-right: auto; width: 100%; }
</style>
</head>
<body class="h-screen bg-slate-950 text-slate-200">
<div class="flex h-screen">

  <!-- ============ PANEL IZQUIERDO (30%) — retractil ============ -->
  <aside id="panelCasos" class="w-[30%] min-w-[300px] max-w-[460px] bg-slate-900 border-r border-slate-800 flex flex-col">

    <!-- VISTA A (vista-repositorio): REPOSITORIO DE CASOS — mutuamente excluyente con #vistaFuentes -->
    <div id="vistaCasos" class="flex flex-col h-full" style="display:flex">
      <div class="px-5 py-4 border-b border-slate-800">
        <h1 class="text-base font-semibold text-slate-100">Mis Casos</h1>
        <p class="text-xs text-slate-400 mt-1">Cada caso agrupa sus propios documentos y su conversación.</p>
      </div>

      <!-- CONSULTA GENERAL: chat global contra el marco normativo (sin caso) -->
      <div class="px-5 py-4 border-b border-slate-800">
        <button id="btnConsultaGeneral" onclick="entrarConsultaGeneral()"
                class="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-500 text-white text-sm font-semibold rounded-md py-2.5 transition ring-1 ring-blue-500/30">
          <span>⚖️</span> Consulta General
        </button>
        <p class="text-[11px] text-slate-500 mt-1.5 flex items-center gap-1">
          <span>🔒</span> Modo estricto: responde solo desde el marco normativo cargado (cero alucinaciones).
        </p>
      </div>

      <div class="px-5 py-4 border-b border-slate-800">
        <label class="block text-xs font-semibold text-slate-300 mb-1">Nuevo caso</label>
        <div class="flex gap-2">
          <input id="nombreCaso" type="text" placeholder="Ej: Licitación carretera MO-108"
                 class="flex-1 bg-slate-800 border border-slate-700 text-slate-100 placeholder-slate-500 rounded-md px-3 py-2 text-sm outline-none focus:border-blue-500"
                 onkeydown="if(event.key==='Enter'){event.preventDefault();crearCaso();}">
        </div>
        <button onclick="crearCaso()"
                class="mt-2 w-full bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium rounded-md py-2 transition">
          + Crear Nuevo Caso
        </button>
      </div>

      <!-- BARRA DE SELECCION MASIVA (Bulk Delete) -->
      <div class="px-4 pt-3 flex items-center justify-between gap-2">
        <span class="text-[11px] text-slate-500"><span id="selCount">0</span> seleccionado(s)</span>
        <button id="btnBulkDel" onclick="eliminarSeleccionados()" disabled
                class="flex items-center gap-1 bg-red-600 hover:bg-red-500 text-white text-xs font-medium rounded-md px-2.5 py-1.5 transition disabled:opacity-30 disabled:cursor-not-allowed">
          🗑 Eliminar Seleccionados
        </button>
      </div>

      <div id="listaCasos" class="flex-1 scroll-y px-4 py-3 space-y-2">
        <p class="text-xs text-slate-500 px-1 py-2">Cargando casos...</p>
      </div>
    </div>

    <!-- VISTA B (vista-detalle-caso): FUENTES DEL CASO — mutuamente excluyente con #vistaCasos -->
    <div id="vistaFuentes" class="flex flex-col h-full" style="display:none">
      <div class="px-5 py-3 border-b border-slate-800">
        <button onclick="volverCasos()" class="text-xs text-blue-400 hover:underline mb-2">&larr; Volver a Mis Casos</button>
        <h1 id="tituloCaso" class="text-base font-semibold text-slate-100 truncate">Fuentes del Caso</h1>
        <p class="text-xs text-slate-400 mt-1">Sube texto, hojas de cálculo, imágenes o ZIP y actívalos con el interruptor para usarlos como contexto.</p>
      </div>

      <div class="px-5 py-4 border-b border-slate-800 space-y-3">
        <div>
          <label class="block text-xs font-semibold text-slate-300 mb-1">Documentos, hojas de cálculo, imágenes o ZIP — puedes elegir varios</label>
          <input id="file" type="file" multiple
                 accept=".pdf,.docx,.doc,.xlsx,.xls,.csv,.png,.jpg,.jpeg,.zip"
                 class="block w-full text-xs text-slate-400 file:mr-3 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-medium file:bg-blue-500/20 file:text-blue-300 hover:file:bg-blue-500/30">
        </div>
        <div>
          <label class="block text-xs font-semibold text-slate-300 mb-1">Etiquetas (Enter o coma)</label>
          <div id="tagbox" class="flex flex-wrap items-center gap-1 bg-slate-800 border border-slate-700 rounded-md px-2 py-1.5">
            <input id="tagin" type="text" placeholder="Ej: Contrato, Ejecucion, Moquegua"
                   class="flex-1 min-w-[120px] bg-transparent text-slate-100 placeholder-slate-500 outline-none text-xs py-0.5">
          </div>
        </div>
        <button id="btnAdd" onclick="agregarFuente()"
                class="w-full bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium rounded-md py-2 transition">
          + Agregar fuente
        </button>
        <p class="text-[11px] text-slate-500">Audita <b>documentos de texto</b> (PDF/Word con OCR automático para escaneos), <b>hojas de cálculo</b> (Excel/CSV → tabla), <b>imágenes de evidencias</b> (PNG/JPG con OCR: series, marcas, notas) y <b>archivos comprimidos</b> (.zip: se desempaqueta y cada archivo entra como fuente independiente).</p>
        <p id="errAdd" class="hidden-x text-xs text-red-300 font-semibold bg-red-500/10 border border-red-500/40 rounded-md px-2 py-1.5"></p>
      </div>

      <div id="lista" class="flex-1 scroll-y px-4 py-3 space-y-2"></div>
      <div class="px-5 py-2 border-t border-slate-800 text-[11px] text-slate-500">
        <span id="contador">0</span> fuente(s) · <span id="activas">0</span> activa(s)
      </div>
    </div>

    <!-- VISTA C (biblioteca-institucional): documentos globales vectorizados -->
    <div id="vistaBiblioteca" class="flex flex-col h-full" style="display:none">
      <div class="px-5 py-3 border-b border-slate-800">
        <button onclick="volverCasos()" class="text-xs text-blue-400 hover:underline mb-2">&larr; Volver a Mis Casos</button>
        <h1 class="text-base font-semibold text-slate-100">📚 Biblioteca Institucional</h1>
        <p class="text-xs text-slate-400 mt-1">Normativa global. Cada documento se fragmenta, se vectoriza y se indexa con su categoría, año y vigencia para la búsqueda híbrida de la Consulta General.</p>
      </div>

      <div class="px-5 py-4 border-b border-slate-800 space-y-3">
        <div>
          <label class="block text-xs font-semibold text-slate-300 mb-1">Documento (.pdf, .docx, .doc, .xlsx, .xls, .csv, imagen)</label>
          <input id="bibFile" type="file" multiple
                 accept=".pdf,.docx,.doc,.xlsx,.xls,.csv,.png,.jpg,.jpeg"
                 class="block w-full text-xs text-slate-400 file:mr-3 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-medium file:bg-blue-500/20 file:text-blue-300 hover:file:bg-blue-500/30">
        </div>
        <div>
          <label class="block text-xs font-semibold text-slate-300 mb-1">Categoría normativa</label>
          <select id="bibCategoria"
                  class="w-full bg-slate-800 border border-slate-700 text-slate-100 rounded-md text-xs px-2 py-1.5 outline-none focus:border-blue-500">
            <optgroup label="MARCO LEGAL Y REGLAMENTARIO" class="bg-slate-900 text-slate-300">
              <option value="leyes_y_reglamentos">Leyes y Reglamentos</option>
            </optgroup>
            <optgroup label="DIRECTIVAS Y LINEAMIENTOS" class="bg-slate-900 text-slate-300">
              <option value="directivas">Directivas y lineamientos</option>
            </optgroup>
            <optgroup label="HERRAMIENTAS Y FORMATOS ESTÁNDAR" class="bg-slate-900 text-slate-300">
              <option value="documentos_orientacion">Documentos de orientación / formatos</option>
            </optgroup>
            <optgroup label="JURISPRUDENCIA Y CRITERIOS VINCULANTES" class="bg-slate-900 text-slate-300">
              <option value="resoluciones_tribunal">Resoluciones del Tribunal</option>
              <option value="opiniones">Opiniones</option>
            </optgroup>
          </select>
        </div>
        <div class="flex items-end gap-3">
          <div class="flex flex-col">
            <label class="text-xs font-semibold text-slate-300 mb-1">Año de emisión</label>
            <input id="bibAnio" type="number" min="1990" max="2100" value="2026"
                   class="w-28 bg-slate-800 border border-slate-700 text-slate-100 rounded-md text-xs px-2 py-1.5 outline-none focus:border-blue-500">
          </div>
          <label class="flex items-center gap-1.5 text-xs text-slate-300 pb-1.5">
            <input type="checkbox" id="bibVigente" checked class="w-4 h-4 accent-blue-500"> Vigente
          </label>
        </div>
        <button id="btnBibAdd" onclick="subirBiblioteca()"
                class="w-full bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium rounded-md py-2 transition">
          + Indexar en la Biblioteca
        </button>
        <p class="text-[11px] text-slate-500">El documento se vectoriza (embeddings) y queda disponible para todas las consultas. Puede tardar según el tamaño.</p>
        <p id="errBib" class="hidden-x text-xs text-red-300 font-semibold bg-red-500/10 border border-red-500/40 rounded-md px-2 py-1.5"></p>
      </div>

      <div class="px-5 pt-2 flex items-center justify-between">
        <span class="text-[11px] text-slate-500">Documentos indexados</span>
        <button onclick="reintentarFallidos()" title="Reprocesar todos los documentos en error"
                class="text-xs text-amber-300 hover:text-amber-200 hover:underline">↻ Reintentar fallidos</button>
      </div>
      <div id="listaBib" class="flex-1 scroll-y px-4 py-3 space-y-2"></div>
    </div>
  </aside>

  <!-- ============ CENTRO: SOLO CHAT ============ -->
  <main class="flex-1 min-w-0 flex flex-col bg-slate-950">
    <header class="flex items-center justify-between gap-3 px-6 py-4 border-b border-slate-800/70">
      <div class="min-w-0">
        <div class="flex items-center gap-2">
          <h2 class="text-base font-semibold text-slate-100">Asistente de contrataciones públicas</h2>
          <span id="badgeEstricto" class="hidden-x items-center gap-1 bg-emerald-500/10 text-emerald-300 ring-1 ring-emerald-400/20 text-[10px] font-medium rounded-full px-2 py-0.5">🔒 Modo estricto</span>
        </div>
        <p class="text-xs text-slate-500 truncate mt-0.5">Caso actual: <span id="casoEnChat">— (ninguno)</span></p>
      </div>
      <div class="flex items-center gap-2 flex-none">
        <button id="btnPanelCasos" onclick="togglePanelCasos()" title="Mostrar u ocultar Mis casos"
                class="inline-flex items-center gap-1.5 border border-slate-700 hover:bg-slate-800 text-slate-300 text-xs font-medium rounded-md px-3 py-1.5 transition">
          ☰ Casos
        </button>
        <button id="btnLimpiar" onclick="limpiarChat()" title="Limpiar chat actual"
                class="inline-flex items-center gap-1.5 border border-slate-700 hover:bg-slate-800 text-slate-300 text-xs font-medium rounded-md px-3 py-1.5 transition disabled:opacity-40 disabled:cursor-not-allowed">
          🧹 Limpiar
        </button>
        <button id="btnPanelNormativo" onclick="togglePanelNormativo()" title="Mostrar u ocultar el marco normativo"
                class="hidden xl:inline-flex items-center gap-1.5 border border-slate-700 hover:bg-slate-800 text-slate-300 text-xs font-medium rounded-md px-3 py-1.5 transition">
          ⚖️ Marco normativo
        </button>
        <div role="group" aria-label="Apariencia" class="inline-flex items-center gap-0.5 border border-slate-700 rounded-md px-1 py-0.5 text-slate-300 text-sm">
          <button class="tema-btn" data-tema="claro" onclick="setTema('claro')" title="Claro" aria-pressed="false">☀️</button>
          <button class="tema-btn" data-tema="sistema" onclick="setTema('sistema')" title="Sistema" aria-pressed="false">🖥️</button>
          <button class="tema-btn" data-tema="oscuro" onclick="setTema('oscuro')" title="Oscuro" aria-pressed="false">🌙</button>
        </div>
      </div>
    </header>

    <div id="chat" class="flex-1 min-h-0 scroll-y px-6 py-6 space-y-4"></div>

    <!-- Input FIJO abajo (la columna de chat ocupa todo el alto; el chat hace scroll propio) -->
    <div class="border-t border-slate-800/70 bg-slate-900/60 px-6 py-3">
      <div class="flex items-end gap-2 max-w-[56rem] mx-auto w-full">
        <textarea id="q" rows="1" placeholder="Entra a un caso para chatear..."
                  class="flex-1 resize-none bg-slate-800 border border-slate-700 text-slate-100 placeholder-slate-500 rounded-lg px-3 py-2 text-sm outline-none focus:border-blue-500 disabled:opacity-50"
                  onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();enviar();}"></textarea>
        <button id="btnChat" onclick="enviar()"
                class="bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium rounded-lg px-4 py-2 transition disabled:opacity-40 disabled:cursor-not-allowed">Analizar</button>
      </div>
      <p id="ayudaChat" class="text-[11px] text-slate-500 mt-1.5 max-w-[56rem] mx-auto w-full">Con una pregunta, responde apoyado en las fuentes del caso y el marco legal. Sin pregunta, analiza las fuentes activas.</p>
    </div>
  </main>

  <!-- ============ DERECHA: MARCO NORMATIVO (colapsable) ============ -->
  <aside id="panelNormativo" class="hidden xl:flex w-80 flex-none flex-col bg-slate-900 border-l border-slate-800">
    <div class="flex items-start justify-between gap-2 px-5 py-4 border-b border-slate-800/70">
      <div class="min-w-0">
        <h2 class="text-sm font-semibold text-slate-100">Marco normativo</h2>
        <p class="text-[11px] text-slate-500 mt-0.5">Filtros y selección de normas (acota el marco legal en consultas y casos).</p>
      </div>
      <button onclick="togglePanelNormativo(false)" title="Ocultar panel"
              class="flex-none text-slate-500 hover:text-slate-200 text-lg leading-none">&times;</button>
    </div>

    <div class="flex-1 min-h-0 overflow-y-auto px-5 py-4">
      <!-- Biblioteca Institucional (movida desde Mis casos al panel derecho) -->
      <button id="btnBiblioteca" onclick="entrarBiblioteca()"
              class="w-full flex items-center justify-center gap-2 bg-slate-700 hover:bg-slate-600 text-slate-100 text-sm font-semibold rounded-md py-2 transition">
        <span>📚</span> Biblioteca Institucional
      </button>
      <p class="text-[11px] text-slate-500 mt-1.5 mb-4">Sube normativa global (se vectoriza con su categoría, año y vigencia para la búsqueda híbrida).</p>

      <!-- FILTROS NORMATIVOS — solo visibles en Consulta General (lo gobierna marcarModoUI) -->
      <div id="filtrosBar" class="flex flex-col gap-5" style="display:none">

        <!-- Categoria / Año / Vigencia -->
        <div class="flex flex-col gap-3">
          <div class="flex flex-col gap-1">
            <label class="text-xs font-medium text-slate-300">Categoría <span class="text-slate-500 font-normal">(multiselección)</span></label>
            <select id="filtroCategorias" multiple size="5"
                    class="bg-slate-800 border border-slate-700 text-slate-200 rounded-md text-xs px-2 py-1.5 outline-none focus:border-blue-500">
              <optgroup label="Marco legal y reglamentario" class="bg-slate-900 text-slate-300">
                <option value="leyes_y_reglamentos">Leyes y reglamentos</option>
              </optgroup>
              <optgroup label="Directivas y lineamientos" class="bg-slate-900 text-slate-300">
                <option value="directivas">Directivas y lineamientos</option>
              </optgroup>
              <optgroup label="Herramientas y formatos estándar" class="bg-slate-900 text-slate-300">
                <option value="documentos_orientacion">Documentos de orientación / formatos</option>
              </optgroup>
              <optgroup label="Jurisprudencia y criterios vinculantes" class="bg-slate-900 text-slate-300">
                <option value="resoluciones_tribunal">Resoluciones del tribunal</option>
                <option value="opiniones">Opiniones</option>
              </optgroup>
            </select>
            <span class="text-[10px] text-slate-500">Sin selección = busca en todas las categorías.</span>
          </div>
          <div class="flex flex-col gap-1">
            <label class="text-xs font-medium text-slate-300">Año de emisión</label>
            <select id="filtroAnio"
                    class="bg-slate-800 border border-slate-700 text-slate-200 rounded-md text-xs px-2 py-1.5 outline-none focus:border-blue-500">
              <option value="Todos">Todos</option>
              <option value="2026">2026</option>
              <option value="2025">2025</option>
              <option value="2024">2024</option>
              <option value="Anteriores">Anteriores</option>
            </select>
          </div>
          <label class="flex items-center gap-2 text-xs text-slate-300">
            <input type="checkbox" id="filtroVigente" checked class="w-4 h-4 accent-blue-500">
            Excluir normativa derogada
          </label>
        </div>

        <!-- Normas individuales -->
        <div class="flex flex-col gap-1.5 border-t border-slate-800/70 pt-4">
          <div class="flex items-center justify-between">
            <label class="text-xs font-medium text-slate-300">Normas</label>
            <span class="text-[11px] text-slate-500">
              <button type="button" onclick="marcarNormas(true)" class="text-blue-400 hover:underline">todas</button> ·
              <button type="button" onclick="marcarNormas(false)" class="text-blue-400 hover:underline">ninguna</button>
            </span>
          </div>
          <div id="filtroNormas"
               class="bg-slate-800/60 border border-slate-700/60 rounded-md text-xs px-2 py-2 max-h-[260px] overflow-y-auto space-y-1.5">
            <p class="text-slate-500 text-[10px]">Cargando normas...</p>
          </div>
          <span class="text-[10px] text-slate-500">Todas activas por defecto; desactiva las que no quieras consultar.</span>
        </div>

      </div>
    </div>
  </aside>

  <!-- PANEL DE CITA (estilo NotebookLM): texto textual del fragmento citado -->
  <div id="citaPanel" class="cita-panel hidden-x">
    <div class="flex items-start justify-between gap-2 px-4 py-3 border-b border-slate-700">
      <div class="min-w-0">
        <div id="citaRef" class="text-xs font-semibold text-slate-100 truncate">—</div>
        <div id="citaMeta" class="text-[11px] text-slate-500 mt-0.5"></div>
      </div>
      <button onclick="cerrarCita()" title="Cerrar" class="flex-none text-slate-500 hover:text-slate-200 text-lg leading-none">&times;</button>
    </div>
    <div class="px-4 py-3">
      <div id="citaTexto" class="cita-quote text-xs"></div>
      <div class="mt-3 text-right">
        <button id="citaFuente" onclick="verFuenteCita()" class="text-[11px] text-blue-400 hover:underline">ver fuente →</button>
      </div>
    </div>
  </div>
</div>

<script>
let casoActual = null;   // {id, nombre}
let modoGeneral = false; // true = Consulta General (chat global sin caso, guardrail estricto)
let fuentes = [];        // fuentes del caso actual
let historial = [];      // memoria multi-turno del caso/consulta actual
const tags = [];
const MAX_HIST_CLIENTE = 40;
// Citas (estilo NotebookLM): mapa cid -> fragmento, para abrir el texto al clic.
let _citas = {};
let _msgSeq = 0;
let _citaActual = null;

// ===== ARQUITECTURA MULTITAREA: estado global desacoplado de la vista activa =====
const seleccion = new Set();   // caso_ids marcados para borrado masivo
const jobs = {};               // caso_id -> Set(jobKey): subidas/OCR en curso (siguen vivos al navegar)
const pollers = new Set();     // fids con poller activo (evita pollers duplicados)
let _tmpSeq = 0;               // contador para claves temporales de subida

function jobAdd(casoId, key){ (jobs[casoId] || (jobs[casoId] = new Set())).add(key); refreshCasoIndicador(casoId); }
function jobDel(casoId, key){ if(jobs[casoId]){ jobs[casoId].delete(key); if(!jobs[casoId].size) delete jobs[casoId]; } refreshCasoIndicador(casoId); }
function jobCount(casoId){ return jobs[casoId] ? jobs[casoId].size : 0; }
function spinnerHTML(txt){
  return '<span class="inline-flex items-center gap-1 text-[10px] text-amber-400 font-semibold">' +
         '<span class="inline-block w-3 h-3 border-2 border-amber-400 border-t-transparent rounded-full animate-spin"></span>' +
         esc(txt) + '</span>';
}
// Actualiza en vivo el indicador "Subiendo..." de un caso en el sidebar (si su tarjeta esta visible).
function refreshCasoIndicador(casoId){
  const el = document.getElementById('spin-' + casoId);
  if(el){ el.innerHTML = jobCount(casoId) ? spinnerHTML('Subiendo...') : ''; }
}

function esc(s){ return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function show(id){ document.getElementById(id).classList.remove('hidden-x'); }
function hide(id){ document.getElementById(id).classList.add('hidden-x'); }

// ===== ALTERNANCIA DE VISTAS DEL SIDEBAR (contenedores mutuamente excluyentes) =====
// Se usa style.display inline (gana al cascade) en vez de la clase .hidden-x, porque
// las utilidades de Tailwind (.flex) se inyectan DESPUES del <style> inline y, al empatar
// en especificidad, sobreescribian a .hidden-x dejando la vista-repositorio apilada.
function mostrarVista(vista){
  // 3 contenedores mutuamente excluyentes (display inline = gana al cascade de Tailwind).
  document.getElementById('vistaCasos').style.display      = (vista === 'repo')       ? 'flex' : 'none';
  document.getElementById('vistaFuentes').style.display    = (vista === 'caso')       ? 'flex' : 'none';
  document.getElementById('vistaBiblioteca').style.display = (vista === 'biblioteca') ? 'flex' : 'none';
}

// Colapsa/expande el panel derecho "Marco normativo" (responsive-safe via clase).
function togglePanelNormativo(mostrar){
  const p = document.getElementById('panelNormativo');
  if(!p) return;
  const ocultar = (mostrar === undefined) ? !p.classList.contains('panel-oculto') : !mostrar;
  p.classList.toggle('panel-oculto', ocultar);
}
// Columna izquierda "Mis casos" retractil (mismo patron que el panel derecho).
function togglePanelCasos(mostrar){
  const p = document.getElementById('panelCasos');
  if(!p) return;
  const ocultar = (mostrar === undefined) ? !p.classList.contains('panel-oculto') : !mostrar;
  p.classList.toggle('panel-oculto', ocultar);
}
// ===== TEMA: claro / oscuro / sistema (persistente) =====
function setTema(modo){
  const dt = {claro:'light', oscuro:'dark', sistema:'system'}[modo] || 'system';
  document.documentElement.setAttribute('data-theme', dt);
  try { localStorage.setItem('tema', modo); } catch(e) {}
  document.querySelectorAll('.tema-btn').forEach(b =>
    b.setAttribute('aria-pressed', b.dataset.tema === modo ? 'true' : 'false'));
}
function initTema(){
  let modo = 'sistema';
  try { modo = localStorage.getItem('tema') || 'sistema'; } catch(e) {}
  setTema(modo);
}

// ===== CITAS desplegables (estilo NotebookLM) =====
function etiquetaTipo(t){
  return ({articulo:'Art.', numeral:'Numeral', opinion:'Opinión', considerando:'Considerando',
           anexo:'Anexo', seccion:'Sección', preambulo:'Preámbulo'})[t] || (t || '');
}
function abrirCita(cid){
  const fr = _citas[cid]; if(!fr) return;
  _citaActual = fr;
  document.getElementById('citaRef').textContent = fr.documento || 'Fuente';
  const refTxt = fr.tipo_referencia
      ? (etiquetaTipo(fr.tipo_referencia) + (fr.referencia ? (' ' + fr.referencia) : ''))
      : (fr.cita || '');
  document.getElementById('citaMeta').textContent =
      [refTxt, fr.fase ? ('fase: ' + fr.fase) : '', fr.emisor || ''].filter(Boolean).join('  ·  ');
  document.getElementById('citaTexto').textContent = fr.texto || '(sin texto del fragmento)';
  document.getElementById('citaPanel').classList.remove('hidden-x');
}
function cerrarCita(){ document.getElementById('citaPanel').classList.add('hidden-x'); }
function verFuenteCita(){
  if(!_citaActual) return;
  togglePanelNormativo(true);                       // asegura visible el panel derecho
  const cont = document.getElementById('filtroNormas');
  const objetivo = (_citaActual.documento || '').trim().toLowerCase();
  if(cont && objetivo){
    for(const l of cont.querySelectorAll('label')){
      if(l.textContent.trim().toLowerCase().includes(objetivo)){
        l.scrollIntoView({block:'center', behavior:'smooth'});
        l.classList.add('cita-flash'); setTimeout(()=>l.classList.remove('cita-flash'), 1500);
        break;
      }
    }
  }
}
// Delegacion: cualquier elemento con data-cid (badge inline o chip) abre su cita.
document.addEventListener('click', function(e){
  const b = e.target.closest('[data-cid]');
  if(b && b.dataset.cid) abrirCita(b.dataset.cid);
});

async function fetchConTimeout(url, opts, ms){
  const ctrl = new AbortController();
  const t = setTimeout(()=>ctrl.abort(), ms);
  try { return await fetch(url, {...opts, signal: ctrl.signal}); }
  finally { clearTimeout(t); }
}
function mostrarErrorAdd(msg){
  const el = document.getElementById('errAdd');
  if(!msg){ el.classList.add('hidden-x'); el.textContent=''; return; }
  el.textContent = msg; el.classList.remove('hidden-x');
}

// =================== REPOSITORIO DE CASOS ===================
async function cargarCasos(){
  const cont = document.getElementById('listaCasos');
  cont.innerHTML = '<p class="text-xs text-slate-400 px-1 py-2">Cargando casos...</p>';
  try{
    const r = await fetch('/api/casos');   // plain fetch: nunca aborta (multitarea)
    const casos = await r.json();
    // Poda la seleccion de casos que ya no existen.
    const idsActuales = new Set((casos||[]).map(c=>c.id));
    [...seleccion].forEach(id=>{ if(!idsActuales.has(id)) seleccion.delete(id); });

    if(!Array.isArray(casos) || !casos.length){
      cont.innerHTML = '<p class="text-xs text-slate-400 px-1 py-2">No hay casos todavía. Crea el primero arriba.</p>';
      actualizarBarraSeleccion();
      return;
    }
    cont.innerHTML = '';
    casos.forEach(c=>{
      const card = document.createElement('div');
      card.className = 'border border-slate-700 bg-slate-800/40 rounded-lg p-3 hover:bg-slate-800 hover:border-blue-600 cursor-pointer flex items-start gap-2';
      const fecha = (c.fecha_creacion||'').replace('T',' ').slice(0,16);
      const chk = '<input type="checkbox" class="mt-1 w-4 h-4 flex-none accent-blue-500 cursor-pointer" '+(seleccion.has(c.id)?'checked':'')+
                  ' title="Seleccionar para borrado masivo" onclick="event.stopPropagation()" onchange="toggleSeleccion(\''+c.id+'\', this.checked)">';
      card.innerHTML =
        chk +
        '<div class="flex-1 min-w-0">' +
          '<div class="text-sm font-semibold text-slate-100 truncate">'+esc(c.nombre)+'</div>' +
          '<div class="text-[10px] text-slate-500 mt-0.5">'+(c.n_fuentes||0)+' fuente(s) · '+esc(fecha)+
            ' <span id="spin-'+c.id+'" class="ml-1 align-middle">'+(jobCount(c.id)?spinnerHTML('Subiendo...'):'')+'</span>' +
          '</div>' +
        '</div>' +
        '<button title="Eliminar caso" class="text-slate-500 hover:text-red-400 text-sm leading-none flex-none">&times;</button>';
      // Entrar al caso al hacer clic en la zona del nombre (no en el checkbox ni en la X).
      card.querySelector('.flex-1').onclick = ()=>entrarCaso(c);
      card.querySelector('button').onclick = (ev)=>{ ev.stopPropagation(); borrarCaso(c); };
      cont.appendChild(card);
    });
    actualizarBarraSeleccion();
  }catch(e){
    cont.innerHTML = '<p class="text-xs text-red-400 px-1 py-2">Error al cargar casos: '+esc(String(e))+'</p>';
  }
}

// ---- Seleccion masiva (Bulk Delete) ----
function toggleSeleccion(id, val){
  if(val) seleccion.add(id); else seleccion.delete(id);
  actualizarBarraSeleccion();
}
function actualizarBarraSeleccion(){
  const n = seleccion.size;
  document.getElementById('selCount').textContent = n;
  document.getElementById('btnBulkDel').disabled = (n === 0);
}
async function eliminarSeleccionados(){
  const ids = [...seleccion];
  if(!ids.length) return;
  if(!confirm('¿Eliminar '+ids.length+' caso(s) seleccionado(s) y TODOS sus documentos? Esta acción no se puede deshacer.')) return;
  try{
    const r = await fetch('/api/casos/bulk', {method:'DELETE', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ids})});
    const d = await r.json().catch(()=>({}));
    if(!r.ok){ alert('No se pudieron eliminar: '+(d.error || ('HTTP '+r.status))); return; }
    // Limpieza de estado: si estabas dentro de un caso borrado -> Consulta General.
    const borraronActual = casoActual && ids.includes(casoActual.id);
    ids.forEach(id=>{ seleccion.delete(id); delete jobs[id]; });
    if(borraronActual){ entrarConsultaGeneral(); }
    cargarCasos();
  }catch(e){ alert('Error al eliminar: '+e); }
}

async function crearCaso(){
  const inp = document.getElementById('nombreCaso');
  const nombre = inp.value.trim();
  if(!nombre){ inp.focus(); return; }
  inp.value = '';                 // libera la UI de inmediato: permite crear Caso A y luego Caso B
  try{
    const r = await fetch('/api/casos', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({nombre})});   // sin AbortController: creacion desacoplada, no bloquea
    if(!r.ok){ alert('No se pudo crear el caso (HTTP '+r.status+').'); return; }
    await cargarCasos();          // aparece silenciosamente en el sidebar, SIN cambiar de vista
  }catch(e){ alert('No se pudo crear el caso: '+e); }
}
async function borrarCaso(c){
  if(!confirm('¿Eliminar el caso "'+c.nombre+'" y todos sus documentos?')) return;
  await fetch('/api/casos/'+c.id, {method:'DELETE'});
  seleccion.delete(c.id); delete jobs[c.id];
  if(casoActual && casoActual.id === c.id){ entrarConsultaGeneral(); }
  cargarCasos();
}

// =================== ENTRAR / SALIR DE UN CASO ===================
function setChat(enabled){
  ['q','btnChat','btnLimpiar'].forEach(id=>document.getElementById(id).disabled = !enabled);
  document.getElementById('q').placeholder = enabled ? 'Escribe tu consulta...' : 'Entra a un caso para chatear...';
}
function resetChatUI(msg){
  historial = [];
  document.getElementById('chat').innerHTML =
    '<div class="flex"><div class="bg-slate-800 border border-slate-700 text-slate-100 rounded-2xl px-4 py-3 max-w-[85%] shadow-sm prosa text-sm">'+msg+'</div></div>';
}
// Refleja en la UI si estamos en Consulta General (modo estricto) o en un caso.
function marcarModoUI(){
  const badge = document.getElementById('badgeEstricto');
  const btnG = document.getElementById('btnConsultaGeneral');
  const ayuda = document.getElementById('ayudaChat');
  // Filtros + seleccion de normas (Biblioteca): se usan en AMBOS modos (Consulta General
  // y dentro de un caso) para acotar la busqueda vectorial. display inline = sin choque de cascade.
  const fb = document.getElementById('filtrosBar');
  if(fb) fb.style.display = 'flex';
  if(modoGeneral){
    badge.classList.remove('hidden-x'); badge.classList.add('inline-flex');
    btnG.classList.add('ring-2','ring-emerald-300');
    ayuda.innerHTML = '🔒 <b>Consulta General (modo estricto):</b> el asistente responde únicamente desde el marco normativo cargado; si no está, lo declara expresamente.';
  } else {
    badge.classList.add('hidden-x'); badge.classList.remove('inline-flex');
    btnG.classList.remove('ring-2','ring-emerald-300');
    ayuda.innerHTML = 'Con una pregunta, responde apoyado en las fuentes del caso y el marco legal. Sin pregunta, analiza las fuentes activas.';
  }
}
async function entrarCaso(c){
  casoActual = c; modoGeneral = false;
  fuentes = []; tags.length = 0; pintarTags(); mostrarErrorAdd('');
  document.getElementById('tituloCaso').textContent = c.nombre;
  document.getElementById('casoEnChat').textContent = c.nombre;
  mostrarVista('caso');   // muestra vista-detalle-caso, oculta vista-repositorio
  setChat(true); marcarModoUI();
  cargarNormas();   // expone la seleccion individual de normas de la Biblioteca tambien en el caso
  resetChatUI('Estás en el caso <b>'+esc(c.nombre)+'</b>. Sube documentos, actívalos y pregúntame. El chat y las fuentes son exclusivos de este caso.');
  // Cargar fuentes del caso (aislado).
  try{
    const r = await fetch('/api/casos/'+c.id+'/fuentes');
    const d = await r.json();
    fuentes = (Array.isArray(d)?d:[]).map(f=>({...f, activo:true}));
    pintarFuentes();
    // Reengancha el seguimiento global de cualquier fuente aun en proceso (sin duplicar pollers).
    fuentes.forEach(f=>{ if((f.estado||'listo')==='procesando'){ jobAdd(c.id, f.id); pollFuenteGlobal(c.id, f.id); } });
  }catch(e){ document.getElementById('lista').innerHTML = '<p class="text-xs text-red-400">Error al cargar fuentes.</p>'; }
}
// CONSULTA GENERAL: chat global contra el marco normativo, sin caso ni fuentes.
function entrarConsultaGeneral(){
  casoActual = null; modoGeneral = true; fuentes = [];
  mostrarVista('repo');  // se mantiene el repositorio visible a la izquierda
  document.getElementById('casoEnChat').textContent = 'Consulta General — marco normativo (Ley 32069)';
  setChat(true); marcarModoUI();
  cargarNormas();   // pobla el menu de seleccion individual de normas
  resetChatUI('Estás en <b>Consulta General</b> (modo estricto 🔒). Pregunta sobre el marco normativo cargado (Ley 32069 y su Reglamento). El asistente responde <b>solo</b> desde la normativa recuperada; si la respuesta no está, lo dirá explícitamente.');
}
function volverCasos(){
  casoActual = null; modoGeneral = false; fuentes = []; historial = [];
  mostrarVista('repo');   // vuelve a vista-repositorio, oculta vista-detalle-caso
  setChat(false); marcarModoUI();
  document.getElementById('casoEnChat').textContent = '— (ninguno)';
  resetChatUI('Selecciona un caso, crea uno nuevo o usa la <b>Consulta General</b> para empezar.');
  cargarCasos();
}

// =================== BIBLIOTECA INSTITUCIONAL ===================
const CAT_LABEL = {
  leyes_y_reglamentos: 'Marco legal y reglamentario',
  directivas: 'Directivas y lineamientos',
  documentos_orientacion: 'Herramientas y formatos',
  resoluciones_tribunal: 'Jurisprudencia (Tribunal)',
  opiniones: 'Jurisprudencia (Opiniones)',
};
function mostrarErrorBib(msg){
  const el = document.getElementById('errBib');
  if(!msg){ el.classList.add('hidden-x'); el.textContent=''; return; }
  el.textContent = msg; el.classList.remove('hidden-x');
}
function entrarBiblioteca(){
  casoActual = null; modoGeneral = false; fuentes = []; mostrarErrorBib('');
  mostrarVista('biblioteca');
  document.getElementById('casoEnChat').textContent = 'Biblioteca Institucional';
  setChat(false); marcarModoUI();
  resetChatUI('📚 <b>Biblioteca Institucional</b>. Sube normativa global: se fragmenta, se vectoriza (embeddings) y se indexa con su <b>categoría, año y vigencia</b>, quedando disponible para la <b>Consulta General</b> y sus filtros normativos.');
  cargarBiblioteca();
}
function tarjetaBib(d){
  const card = document.createElement('div');
  const estado = d.estado || 'listo';
  card.className = 'border rounded-lg p-3 ' +
    (estado==='error' ? 'bg-red-500/10 border-red-500/40' : 'bg-slate-800/40 border-slate-700');
  let est;
  if(estado==='procesando'){
    est = '<div class="text-[10px] text-amber-400 font-semibold flex items-center gap-1">' +
          '<span class="inline-block w-3 h-3 border-2 border-amber-400 border-t-transparent rounded-full animate-spin"></span>' +
          'Vectorizando e indexando...</div>';
  } else if(estado==='error'){
    est = '<div class="text-[10px] text-red-400 font-semibold">Error: '+esc(d.error||'no se pudo indexar')+'</div>';
  } else {
    const m = {ocr:'OCR', tabla:'TABLA'}[d.metodo];
    est = '<div class="text-[10px] text-green-400 font-semibold">'+(d.n_chunks||0)+' vectores'+(m?' · '+m:'')+'</div>';
  }
  const cat = esc(CAT_LABEL[d.categoria] || d.categoria || '-');
  const vig = d.vigente ? '<span class="text-green-400">vigente</span>' : '<span class="text-red-400">derogada</span>';
  const reint = (estado==='error')
    ? '<button data-retry title="Reprocesar este documento" class="text-[10px] text-blue-400 hover:underline whitespace-nowrap">↻ Reintentar</button>'
    : '';
  card.innerHTML =
    '<div class="flex items-start gap-2">' +
      '<div class="flex-1 min-w-0">' +
        '<div class="text-xs font-semibold text-slate-100 truncate" title="'+esc(d.nombre)+'">'+esc(d.nombre)+'</div>' +
        '<div class="text-[10px] text-slate-500 mt-0.5">'+cat+' · '+(d.anio||'s/año')+' · '+vig+'</div>' +
        est +
      '</div>' +
      '<div class="flex flex-col items-end gap-1 flex-none">' + reint +
        '<button data-del title="Eliminar del vector store" class="text-slate-500 hover:text-red-400 text-sm leading-none">&times;</button>' +
      '</div>' +
    '</div>';
  card.querySelector('[data-del]').onclick = ()=>borrarBiblioteca(d.id, d.nombre);
  const rb = card.querySelector('[data-retry]');
  if(rb) rb.onclick = ()=>reintentarBiblio(d.id);
  return card;
}
async function reintentarBiblio(id){
  try{
    const r = await fetch('/api/biblioteca/'+id+'/reintentar', {method:'POST'});
    const d = await r.json().catch(()=>({}));
    if(!r.ok){ mostrarErrorBib(d.error || ('No se pudo reintentar (HTTP '+r.status+').')); return; }
    mostrarErrorBib(''); cargarBiblioteca();
  }catch(e){ mostrarErrorBib('Error al reintentar: '+(e && e.message ? e.message : e)); }
}
async function reintentarFallidos(){
  try{
    const r = await fetch('/api/biblioteca/reintentar-fallidos', {method:'POST'});
    const d = await r.json().catch(()=>({}));
    if(!r.ok){ mostrarErrorBib(d.error || ('Error (HTTP '+r.status+').')); return; }
    if(!d.reintentados && !d.sin_copia){ mostrarErrorBib('No hay documentos en error para reintentar.'); return; }
    let msg = 'Reintentando '+d.reintentados+' documento(s).';
    if(d.sin_copia) msg += ' '+d.sin_copia+' sin copia guardada (re-súbelos).';
    mostrarErrorBib(d.sin_copia ? msg : '');
    cargarBiblioteca();
  }catch(e){ mostrarErrorBib('Error: '+(e && e.message ? e.message : e)); }
}
async function cargarBiblioteca(){
  const cont = document.getElementById('listaBib');
  cont.innerHTML = '<p class="text-xs text-slate-500 px-1 py-2">Cargando biblioteca...</p>';
  try{
    const r = await fetch('/api/biblioteca');
    const docs = await r.json();
    if(!Array.isArray(docs) || !docs.length){
      cont.innerHTML = '<p class="text-xs text-slate-500 px-1 py-2">Aún no hay documentos institucionales indexados.</p>'; return;
    }
    cont.innerHTML = '';
    docs.forEach(d=>{ cont.appendChild(tarjetaBib(d)); if((d.estado||'')==='procesando') pollBiblioteca(d.id); });
  }catch(e){ cont.innerHTML = '<p class="text-xs text-red-400 px-1 py-2">Error al cargar la biblioteca.</p>'; }
}
function subirBiblioteca(){
  mostrarErrorBib('');
  const input = document.getElementById('bibFile');
  const archivos = Array.from(input.files || []);
  if(!archivos.length){ mostrarErrorBib('Selecciona uno o más documentos.'); return; }
  const categoria = document.getElementById('bibCategoria').value;
  const anio = parseInt(document.getElementById('bibAnio').value||'0', 10) || 0;
  const vigente = document.getElementById('bibVigente').checked;
  input.value = '';
  archivos.forEach(f => subirUnBiblio(f, categoria, anio, vigente));
}
async function subirUnBiblio(file, categoria, anio, vigente){
  const fd = new FormData();
  fd.append('archivo', file); fd.append('categoria', categoria);
  fd.append('anio', String(anio)); fd.append('vigente', vigente ? 'true' : 'false');
  try{
    const r = await fetch('/api/biblioteca', {method:'POST', body:fd});   // sin abort
    let d = {}; try { d = await r.json(); } catch(_){}
    if(!r.ok || d.error){ mostrarErrorBib('"'+file.name+'": '+(d.error || ('error '+r.status))); return; }
    cargarBiblioteca();
    if((d.estado||'')==='procesando') pollBiblioteca(d.id);
  }catch(e){ mostrarErrorBib('No se pudo subir "'+file.name+'": '+(e && e.message ? e.message : e)); }
}
function pollBiblioteca(bid){
  let intentos = 0;
  const iv = setInterval(async ()=>{
    intentos++;
    if(intentos > 240){ clearInterval(iv); return; }   // ~12 min (los embeddings pueden tardar)
    try{
      const r = await fetch('/api/biblioteca/'+bid);
      if(r.status===404){ clearInterval(iv); cargarBiblioteca(); return; }
      if(!r.ok) return;
      const d = await r.json();
      if((d.estado||'') !== 'procesando'){ clearInterval(iv); cargarBiblioteca(); }
    }catch(e){ /* reintenta */ }
  }, 3000);
}
async function borrarBiblioteca(bid, nombre){
  if(!confirm('¿Eliminar "'+nombre+'" de la Biblioteca? Se borrarán sus vectores en BigQuery.')) return;
  try{
    const r = await fetch('/api/biblioteca/'+bid, {method:'DELETE'});
    if(!r.ok){ const d = await r.json().catch(()=>({})); alert('No se pudo eliminar: '+(d.error||('HTTP '+r.status))); return; }
    cargarBiblioteca();
  }catch(e){ alert('Error al eliminar: '+e); }
}

// =================== ETIQUETAS (chips) ===================
const tagbox = document.getElementById('tagbox'), tagin = document.getElementById('tagin');
function pintarTags(){
  tagbox.querySelectorAll('.tagchip').forEach(e=>e.remove());
  tags.forEach((t,i)=>{
    const c=document.createElement('span');
    c.className='tagchip bg-blue-600 text-white rounded-full px-2 py-0.5 text-xs flex items-center gap-1';
    c.innerHTML=esc(t)+' <span class="cursor-pointer font-bold" onclick="quitarTag('+i+')">&times;</span>';
    tagbox.insertBefore(c, tagin);
  });
}
function agregarTag(v){ v=v.trim().replace(/,$/,'').trim(); if(v && !tags.includes(v)){ tags.push(v); pintarTags(); } }
function quitarTag(i){ tags.splice(i,1); pintarTags(); }
tagin.addEventListener('keydown', e=>{
  if(e.key==='Enter'||e.key===','){ e.preventDefault(); agregarTag(tagin.value); tagin.value=''; }
  else if(e.key==='Backspace' && !tagin.value && tags.length){ quitarTag(tags.length-1); }
});

// =================== FUENTES DEL CASO ===================
function contar(){
  document.getElementById('contador').textContent = fuentes.length;
  document.getElementById('activas').textContent = fuentes.filter(f=>f.activo && (f.estado||'listo')==='listo').length;
}
function pintarFuentes(){
  const cont = document.getElementById('lista');
  if(!fuentes.length){ cont.innerHTML = '<p class="text-xs text-slate-500 px-1 py-2">Aún no hay fuentes en este caso.</p>'; contar(); return; }
  cont.innerHTML = '';
  fuentes.forEach(f=>{
    const estado = f.estado || 'listo';
    const listo = estado === 'listo';
    const card = document.createElement('div');
    card.className = 'border rounded-lg p-3 ' +
      (estado==='error' ? 'bg-red-500/10 border-red-500/40'
        : (listo && f.activo) ? 'bg-blue-500/10 border-blue-600/50'
        : 'bg-slate-800/50 border-slate-700');
    const tagsHtml = (f.etiquetas||[]).map(t=>'<span class="bg-slate-700 text-slate-300 rounded px-1.5 py-0.5 text-[10px]">'+esc(t)+'</span>').join(' ');
    const dis = listo ? '' : 'disabled';
    const toggle = '<label class="switch mt-0.5"><input type="checkbox" '+(f.activo&&listo?'checked':'')+' '+dis+
                   ' onchange="toggleFuente(\''+f.id+'\', this.checked)"><span class="slider"></span></label>';
    let estadoHtml;
    if(estado==='procesando'){
      estadoHtml = '<div class="text-[10px] text-amber-600 font-semibold mb-1 flex items-center gap-1">' +
                   '<span class="inline-block w-3 h-3 border-2 border-amber-500 border-t-transparent rounded-full animate-spin"></span>' +
                   'Procesando (extracción / OCR)...</div>';
    } else if(estado==='error'){
      estadoHtml = '<div class="text-[10px] text-red-400 font-semibold mb-1">Error: '+esc(f.error||'no se pudo procesar')+'</div>';
    } else {
      const etiq = {ocr:'OCR', tabla:'TABLA', zip:'ZIP'}[f.metodo];
      const badge = etiq ? '<span class="text-[10px] text-green-400 font-semibold">'+etiq+'</span>' : '';
      estadoHtml = '<div class="text-[10px] text-slate-500 mb-1">'+(f.n_chars||0).toLocaleString()+' chars '+badge+'</div>';
    }
    card.innerHTML =
      '<div class="flex items-start gap-2">' + toggle +
        '<div class="flex-1 min-w-0">' +
          '<div class="text-xs font-semibold text-slate-100 truncate" title="'+esc(f.nombre)+'">'+esc(f.nombre)+'</div>' +
          estadoHtml +
          '<div class="flex flex-wrap gap-1">'+tagsHtml+'</div>' +
        '</div>' +
        '<button onclick="borrarFuente(\''+f.id+'\')" class="text-slate-500 hover:text-red-400 text-sm leading-none">&times;</button>' +
      '</div>';
    cont.appendChild(card);
  });
  contar();
}
function toggleFuente(id, val){ const f=fuentes.find(x=>x.id===id); if(f){ f.activo=val; } contar(); pintarFuentes(); }

// Poller GLOBAL: sigue corriendo aunque el usuario cambie de caso o vuelva al repositorio.
// No esta atado a casoActual; solo refresca la tarjeta en vivo si ese caso esta abierto.
function pollFuenteGlobal(casoId, fid){
  if(pollers.has(fid)) return;          // ya hay un poller para esta fuente
  pollers.add(fid);
  let intentos = 0;
  const iv = setInterval(async ()=>{
    intentos++;
    if(intentos > 120){   // ~5 min
      clearInterval(iv); pollers.delete(fid); jobDel(casoId, fid);
      if(casoActual && casoActual.id===casoId){ const f=fuentes.find(x=>x.id===fid); if(f){ f.estado='error'; f.error='Tiempo de espera agotado (5 min).'; pintarFuentes(); } }
      return;
    }
    try{
      const r = await fetch('/api/casos/'+casoId+'/fuentes/'+fid);
      if(r.status===404){ clearInterval(iv); pollers.delete(fid); jobDel(casoId, fid); return; }
      if(!r.ok) return;
      const d = await r.json();
      if(casoActual && casoActual.id===casoId){   // refresco en vivo solo si ese caso esta abierto
        const f = fuentes.find(x=>x.id===fid);
        if(f){ f.estado=d.estado; f.n_chars=d.n_chars; f.metodo=d.metodo; f.error=d.error; pintarFuentes(); }
      }
      if(d.estado!=='procesando'){
        clearInterval(iv); pollers.delete(fid); jobDel(casoId, fid);
        // Un ZIP se expandio en fuentes hijas: recarga la lista del caso para mostrarlas.
        if(d.metodo==='zip' && casoActual && casoActual.id===casoId){ recargarFuentesCaso(casoId); }
      }
    }catch(e){ /* reintenta */ }
  }, 2500);
}
// Recarga las fuentes del caso desde el servidor (p. ej. tras expandir un ZIP),
// preservando el estado activo/inactivo de las fuentes ya conocidas.
async function recargarFuentesCaso(casoId){
  if(!casoActual || casoActual.id!==casoId) return;
  try{
    const r = await fetch('/api/casos/'+casoId+'/fuentes');
    const d = await r.json();
    const previos = new Map(fuentes.map(f=>[f.id, f.activo]));
    fuentes = (Array.isArray(d)?d:[]).map(f=>({...f, activo: previos.has(f.id) ? previos.get(f.id) : true}));
    pintarFuentes();
    fuentes.forEach(f=>{ if((f.estado||'listo')==='procesando'){ jobAdd(casoId, f.id); pollFuenteGlobal(casoId, f.id); } });
  }catch(e){ /* sin-op */ }
}
async function borrarFuente(id){
  if(!casoActual) return;
  await fetch('/api/casos/'+casoActual.id+'/fuentes/'+id, {method:'DELETE'});
  fuentes = fuentes.filter(x=>x.id!==id); pintarFuentes();
}
// Subida MULTIPLE: recorre todos los archivos seleccionados y dispara una
// subida asincrona INDEPENDIENTE por cada uno (cada una con su tarjeta y polling).
function agregarFuente(){
  mostrarErrorAdd('');
  if(!casoActual){ mostrarErrorAdd('Entra a un caso primero.'); return; }
  if(tagin.value){ agregarTag(tagin.value); tagin.value=''; }
  const input = document.getElementById('file');
  const archivos = Array.from(input.files || []);
  if(!archivos.length){ mostrarErrorAdd('Selecciona uno o más archivos (PDF, Word, Excel/CSV, imagen o ZIP).'); return; }

  // Las etiquetas actuales se aplican a todo el lote.
  const etiquetasLote = tags.slice();
  const casoId = casoActual.id;

  // LIMPIEZA DEL FORMULARIO: listo de inmediato para una nueva subida.
  input.value = '';
  tags.length = 0; pintarTags();

  // Dispara cada subida de forma concurrente (no se espera una por una).
  archivos.forEach(f => subirUnArchivo(casoId, f, etiquetasLote));
}

// Sube UN archivo DESACOPLADO de la vista: plain fetch (no AbortController), por lo que
// cambiar de caso NO cancela la subida; el seguimiento vive en `jobs` (global).
async function subirUnArchivo(casoId, file, etiquetasArr){
  const fd = new FormData();
  fd.append('archivo', file);
  fd.append('etiquetas', etiquetasArr.join(','));
  const tmpKey = 'tmp_' + (++_tmpSeq);   // marca el job mientras viaja el POST (aun sin fid)
  jobAdd(casoId, tmpKey);
  try{
    const r = await fetch('/api/casos/'+casoId+'/fuentes', {method:'POST', body:fd});  // SIN abort
    let d = {}; try { d = await r.json(); } catch(_){ d = {}; }
    jobDel(casoId, tmpKey);
    if(!r.ok || d.error){
      const msg = '"'+file.name+'": '+(d.error || ('error '+r.status));
      if(casoActual && casoActual.id===casoId) mostrarErrorAdd(msg);
      return;
    }
    // Si el caso sigue abierto, agrega su tarjeta en vivo; si no, sigue en segundo plano.
    if(casoActual && casoActual.id===casoId){ d.activo = true; fuentes.push(d); pintarFuentes(); }
    if((d.estado||'listo')==='procesando'){ jobAdd(casoId, d.id); pollFuenteGlobal(casoId, d.id); }
  }catch(e){
    jobDel(casoId, tmpKey);
    if(casoActual && casoActual.id===casoId){ mostrarErrorAdd('No se pudo subir "'+file.name+'": ' + (e && e.message ? e.message : e)); }
  }
}

// =================== CHAT ===================
const chat = document.getElementById('chat');
function limpiarChat(){
  if(!casoActual && !modoGeneral) return;
  resetChatUI(modoGeneral
    ? 'Consulta General reiniciada (modo estricto 🔒).'
    : 'Chat reiniciado para el caso <b>'+esc(casoActual.nombre)+'</b>.');
}
function addMsg(html, lado){
  const wrap = document.createElement('div'); wrap.className = 'flex ' + (lado==='user'?'justify-end':'');
  const b = document.createElement('div');
  b.className = (lado==='user'
    ? 'bg-blue-600 text-white rounded-2xl px-4 py-3 max-w-[85%] shadow-sm prosa text-sm'
    : 'bg-slate-800 border border-slate-700 text-slate-100 rounded-2xl px-4 py-3 max-w-[85%] shadow-sm prosa text-sm');
  b.innerHTML = html; wrap.appendChild(b); chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight; return b;
}
// Carga las normas (corpus + Biblioteca) en el menu de seleccion individual.
async function cargarNormas(){
  const cont = document.getElementById('filtroNormas');
  cont.innerHTML = '<p class="text-slate-500 text-[10px]">Cargando normas...</p>';
  try{
    const r = await fetch('/api/normas');
    const normas = await r.json();
    if(!Array.isArray(normas) || !normas.length){
      cont.innerHTML = '<p class="text-slate-500 text-[10px]">No hay normas indexadas.</p>'; return;
    }
    cont.innerHTML = '';
    normas.forEach(n=>{
      const lab = document.createElement('label');
      lab.className = 'flex items-center gap-1.5 text-slate-200';
      const cb = document.createElement('input');
      cb.type = 'checkbox'; cb.className = 'w-3.5 h-3.5 accent-blue-500 flex-none';
      cb.checked = true; cb.value = n.doc_id;     // value por propiedad (sin riesgo de inyeccion)
      const txt = document.createElement('span');
      txt.className = 'truncate'; txt.title = n.label; txt.textContent = n.label;
      const meta = document.createElement('span');
      meta.className = 'text-slate-500 text-[10px] ml-auto whitespace-nowrap pl-2';
      meta.textContent = [(CAT_LABEL[n.categoria] || n.categoria || ''), (n.anio || 's/año')]
        .filter(Boolean).join(' · ');
      lab.append(cb, txt, meta);
      cont.appendChild(lab);
    });
  }catch(e){ cont.innerHTML = '<p class="text-red-400 text-[10px]">Error al cargar normas.</p>'; }
}
function marcarNormas(val){
  document.querySelectorAll('#filtroNormas input[type=checkbox]').forEach(i=>{ i.checked = val; });
}

// Lee los Filtros Normativos de la UI (Busqueda Hibrida / pre-filtering).
function leerFiltros(){
  const sel = document.getElementById('filtroCategorias');
  const cats = sel ? Array.from(sel.selectedOptions).map(o=>o.value) : [];
  // Seleccion individual de normas (Biblioteca): aplica en AMBOS modos (caso y general),
  // si ya se cargaron las normas. null = sin restriccion (robusto); lista = solo esas;
  // [] = ninguna seleccionada (busca en cero normas).
  const ni = document.querySelectorAll('#filtroNormas input[type=checkbox]');
  let normas = null;
  if(ni.length){
    normas = Array.from(ni).filter(i=>i.checked).map(i=>i.value);
  }
  return {
    categorias: cats,
    excluir_derogada: document.getElementById('filtroVigente').checked,
    anio: document.getElementById('filtroAnio').value,
    normas: normas,
  };
}
async function enviar(){
  if(!casoActual && !modoGeneral){ return; }
  const q = document.getElementById('q');
  const texto = q.value.trim();
  const activas = modoGeneral ? [] : fuentes.filter(f=>f.activo && (f.estado||'listo')==='listo').map(f=>f.id);
  // Guards: en general se exige pregunta; en un caso, pregunta O al menos una fuente activa.
  if(modoGeneral && !texto){ return; }
  if(!modoGeneral && !texto && !activas.length){
    addMsg('Escribe una consulta o activa al menos una fuente del caso.', 'bot'); return;
  }

  const userTxt = texto;   // sin pregunta = analisis libre de las fuentes (fallback del backend)
  if(texto) addMsg(esc(texto), 'user');
  else addMsg('<i>Analizar las fuentes activas</i>', 'user');
  q.value='';
  document.getElementById('btnChat').disabled=true;
  const cargandoMsg = modoGeneral ? 'Consultando el marco normativo (modo estricto)...' : 'Analizando con las fuentes activas y la normativa...';
  const cargando = addMsg('<span class="text-slate-400 italic">'+cargandoMsg+'</span>', 'bot');
  try{
    const r = await fetchConTimeout('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({caso_id: casoActual ? casoActual.id : '', general: modoGeneral,
                            pregunta: texto, fuentes_activas: activas, modo: 'chat', k: 5,
                            historial: historial.slice(-MAX_HIST_CLIENTE),
                            filtros: leerFiltros()})}, 180000);
    let d = {}; try { d = await r.json(); } catch(_){ d = {}; }
    if(!r.ok){ cargando.innerHTML = '<span class="text-amber-700">Error '+r.status+': '+esc(d.error||'fallo del servidor')+'</span>'; return; }
    if(d.error){ cargando.innerHTML = '<span class="text-amber-700">'+esc(d.error)+'</span>'; return; }
    _msgSeq++;
    const _frags = d.fuentes_normativas || [];
    let h = esc(d.respuesta || 'Sin respuesta.');
    // Marcadores [N] -> badge cobre clicable (split/join: sin regex ni escapes).
    _frags.forEach(fr => {
      if(fr.numero == null) return;
      const cid = _msgSeq + '_' + fr.numero; _citas[cid] = fr;
      const badge = '<button type="button" class="cita-badge" data-cid="'+cid+'" title="Ver fuente">'+fr.numero+'</button>';
      h = h.split('['+fr.numero+']').join(badge);
    });
    if(d.fuentes_usadas && d.fuentes_usadas.length){
      h += '<div class="mt-2 pt-2 border-t border-slate-700 text-[11px] text-slate-400"><b>Fuentes del caso usadas:</b> '
         + d.fuentes_usadas.map(x=>esc(x.nombre)).join(', ') + '</div>';
    }
    if(_frags.length){
      h += '<div class="mt-2 pt-2 border-t border-slate-700 text-[11px] text-slate-400"><b>Fuentes citadas:</b><br>'
         + _frags.map(s=>'<button type="button" data-cid="'+(_msgSeq+'_'+s.numero)+'" class="inline-flex items-center gap-1 bg-slate-700 border border-slate-600 text-slate-300 rounded px-1.5 py-0.5 mt-1 mr-1 hover:border-blue-500 transition"><span class="cita-badge">'+s.numero+'</span>'+esc(s.cita || s.documento)+'</button>').join('') + '</div>';
    }
    cargando.innerHTML = h;
    if(userTxt) historial.push({rol:'user', texto:userTxt});
    historial.push({rol:'model', texto: d.respuesta || ''});
    if(historial.length > MAX_HIST_CLIENTE) historial = historial.slice(-MAX_HIST_CLIENTE);
  }catch(e){
    if(e && e.name==='AbortError'){ cargando.innerHTML = '<span class="text-amber-700">La consulta tardó demasiado y se canceló (timeout).</span>'; }
    else { cargando.innerHTML = '<span class="text-amber-700">Error: '+esc(String(e))+'</span>'; }
  }
  finally{ document.getElementById('btnChat').disabled=false; chat.scrollTop=chat.scrollHeight; }
}

// =================== INIT ===================
initTema();             // aplica tema guardado (claro/oscuro/sistema) y marca el boton activo
mostrarVista('repo');   // estado inicial: vista-repositorio visible, detalle-caso oculto
setChat(false); marcarModoUI();
resetChatUI('Bienvenido. Usa <b>Consulta General</b> 🔒 para preguntar sobre el marco normativo, o entra a un caso para trabajar con tus propios documentos.');
cargarCasos();
</script>
</body>
</html>
"""


# Materializa el frontend en disco (static/index.html) para servirlo con FileResponse /
# StaticFiles. Fuente unica = el string HTML de arriba; el archivo se regenera al arrancar.
try:
    with open(INDEX_HTML_PATH, "w", encoding="utf-8") as _fh:
        _fh.write(HTML)
except OSError as _e:
    print(f"[arranque] no se pudo materializar index.html: {_e}")
