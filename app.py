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

import uuid
import json
import sqlite3
import threading
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from google.genai.types import GenerateContentConfig, Content, Part

# Motor RAG ya implementado y probado.
from responder import recuperar, construir_contexto, cliente, MODELO_GEN
# Extraccion de texto (en memoria) con FALLBACK de OCR (Cloud Vision).
from extraccion_texto import extraer_pdf_inteligente, extraer_docx

app = FastAPI(title="Asistente RAG - Repositorio de Casos")

# ===================== CONFIGURACION =====================
DB_PATH = "casos.db"
_db_lock = threading.Lock()

MAX_CONTEXT_CHARS = 200_000        # tope de texto de fuentes activas enviado al LLM
QUERY_DOC_CHARS = 3_000            # extracto para construir la consulta de recuperacion
MAX_TURNOS_HISTORIAL = 12          # turnos de historial inyectados al modelo
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


_init_db()  # crea la base/tablas al cargar el modulo


# ============================ MODELOS ============================
class NuevoCaso(BaseModel):
    nombre: str = ""


class Turno(BaseModel):
    rol: str = "user"
    texto: str = ""


class Mensaje(BaseModel):
    caso_id: str = ""
    pregunta: str = ""
    fuentes_activas: list[str] = []
    modo: str = "chat"
    k: int = 5
    historial: list[Turno] = []


# ============================ HELPERS RAG ============================
def _doc_label(documento: str) -> str:
    return "Ley" if "ley-general" in documento else "Reglamento"


def _fuentes_normativas(filas):
    return [{
        "documento": _doc_label(f["documento"]),
        "articulo_num": f["articulo_num"],
        "articulo_titulo": f["articulo_titulo"],
        "relevancia": round(1 - f["distance"], 3),
    } for f in filas]


def _extraer_texto_temporal(nombre: str, data: bytes):
    """Extrae texto plano EN MEMORIA (PDF con fallback OCR). Devuelve (texto, metodo)."""
    lower = (nombre or "").lower()
    if lower.endswith(".pdf"):
        texto, _, metodo = extraer_pdf_inteligente(data)
        return texto, metodo
    if lower.endswith(".docx"):
        texto, _ = extraer_docx(data)
        return texto, "nativo"
    raise ValueError("Formato no soportado. Solo se admiten .pdf o .docx.")


def _procesar_fuente(fid, nombre, data):
    """Worker en segundo plano: extrae texto (con OCR si aplica) y actualiza el estado en la BD."""
    try:
        texto, metodo = _extraer_texto_temporal(nombre, data)
        if not texto or len(texto.strip()) < 20:
            actualizar_fuente(fid, estado="error",
                              error="No se pudo extraer texto del documento ni con OCR (vacio o danado).")
            return
        actualizar_fuente(fid, texto=texto, n_chars=len(texto), metodo=metodo, estado="listo", error=None)
    except ValueError as e:
        actualizar_fuente(fid, estado="error", error=str(e))
    except Exception as e:
        actualizar_fuente(fid, estado="error", error=f"{type(e).__name__}: {e}")


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
        if not (lower.endswith(".pdf") or lower.endswith(".docx")):
            return JSONResponse(status_code=400, content={
                "error": "Formato no soportado. Solo se admiten .pdf o .docx."})

        data = await archivo.read()
        if not data:
            return JSONResponse(status_code=400, content={"error": "El archivo llego vacio."})

        fid = crear_fuente_registro(cid, archivo.filename, lista_etiquetas)
        background_tasks.add_task(_procesar_fuente, fid, archivo.filename, data)
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


# ---- Chat / Analisis (aislado por caso) ----
@app.post("/api/chat")
def chat(m: Mensaje):
    pregunta = (m.pregunta or "").strip()

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
    base_query = pregunta or (", ".join(etiquetas))
    consulta = (base_query + "\n" + contexto_fuentes[:QUERY_DOC_CHARS]).strip()
    filas = recuperar(consulta, k=m.k) if consulta else []
    contexto_normas = construir_contexto(filas) if filas else "(sin normas recuperadas)"

    secciones = [f"NORMAS RECUPERADAS (base vectorial):\n{contexto_normas}"]
    if contexto_fuentes:
        secciones.append("DOCUMENTOS DEL CASO (fuentes activas seleccionadas por el usuario):\n"
                         + contexto_fuentes)
    else:
        secciones.append("DOCUMENTOS DEL CASO: (ninguna fuente activa en este turno).")

    if m.modo == "analisis":
        sistema = _sistema_auditoria(etiquetas)
        secciones.append("TAREA: Realiza la auditoria legal de las fuentes activas segun las "
                         "instrucciones del sistema." +
                         (f"\nFoco adicional del usuario: {pregunta}" if pregunta else ""))
    else:
        sistema = _sistema_chat()
        secciones.append(f"CONSULTA DEL USUARIO:\n{pregunta or '(resume y comenta las fuentes activas)'}")

    prompt = "\n\n".join(secciones)

    # MEMORIA MULTI-TURNO: historial previo como turnos nativos user/model.
    contents = []
    for t in m.historial[-MAX_TURNOS_HISTORIAL:]:
        rol = "model" if (t.rol or "").lower() in ("model", "assistant", "ia", "bot") else "user"
        txt = (t.texto or "").strip()
        if txt:
            contents.append(Content(role=rol, parts=[Part(text=txt)]))
    while contents and contents[0].role == "model":
        contents.pop(0)
    contents.append(Content(role="user", parts=[Part(text=prompt)]))

    resp = cliente().models.generate_content(
        model=MODELO_GEN, contents=contents,
        config=GenerateContentConfig(system_instruction=sistema, temperature=0.2),
    )

    return {
        "respuesta": resp.text,
        "modo": m.modo,
        "fuentes_usadas": [{"id": f["id"], "nombre": f["nombre"]} for f in activos],
        "fuentes_normativas": _fuentes_normativas(filas),
    }


# ============================== UI =============================
@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


HTML = r"""
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Asistente de Contrataciones Públicas — Repositorio de Casos</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  .scroll-y { overflow-y: auto; }
  .switch { position: relative; display: inline-block; width: 38px; height: 22px; flex: none; }
  .switch input { opacity: 0; width: 0; height: 0; }
  .slider { position: absolute; cursor: pointer; inset: 0; background: #cbd5e1; border-radius: 9999px; transition: .2s; }
  .slider:before { content: ""; position: absolute; height: 16px; width: 16px; left: 3px; top: 3px; background: #fff; border-radius: 9999px; transition: .2s; }
  input:checked + .slider { background: #2563eb; }
  input:checked + .slider:before { transform: translateX(16px); }
  .prosa { white-space: pre-wrap; line-height: 1.6; }
  .hidden-x { display: none; }
</style>
</head>
<body class="h-screen bg-slate-100 text-slate-800">
<div class="flex h-screen">

  <!-- ============ PANEL IZQUIERDO (30%) ============ -->
  <aside class="w-[30%] min-w-[300px] max-w-[460px] bg-white border-r border-slate-200 flex flex-col">

    <!-- VISTA A: REPOSITORIO DE CASOS -->
    <div id="vistaCasos" class="flex flex-col h-full">
      <div class="px-5 py-4 border-b border-slate-200">
        <h1 class="text-base font-semibold text-slate-900">Mis Casos</h1>
        <p class="text-xs text-slate-500 mt-1">Cada caso agrupa sus propios documentos y su conversación.</p>
      </div>
      <div class="px-5 py-4 border-b border-slate-200">
        <label class="block text-xs font-semibold text-slate-600 mb-1">Nuevo caso</label>
        <div class="flex gap-2">
          <input id="nombreCaso" type="text" placeholder="Ej: Licitación carretera MO-108"
                 class="flex-1 border border-slate-300 rounded-md px-3 py-2 text-sm outline-none focus:border-blue-500"
                 onkeydown="if(event.key==='Enter'){event.preventDefault();crearCaso();}">
        </div>
        <button onclick="crearCaso()"
                class="mt-2 w-full bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-md py-2 transition">
          + Crear Nuevo Caso
        </button>
      </div>
      <div id="listaCasos" class="flex-1 scroll-y px-4 py-3 space-y-2">
        <p class="text-xs text-slate-400 px-1 py-2">Cargando casos...</p>
      </div>
    </div>

    <!-- VISTA B: FUENTES DEL CASO -->
    <div id="vistaFuentes" class="hidden-x flex-col h-full">
      <div class="px-5 py-3 border-b border-slate-200">
        <button onclick="volverCasos()" class="text-xs text-blue-700 hover:underline mb-2">&larr; Volver a Mis Casos</button>
        <h1 id="tituloCaso" class="text-base font-semibold text-slate-900 truncate">Fuentes del Caso</h1>
        <p class="text-xs text-slate-500 mt-1">Sube documentos y actívalos con el interruptor para usarlos como contexto.</p>
      </div>

      <div class="px-5 py-4 border-b border-slate-200 space-y-3">
        <div>
          <label class="block text-xs font-semibold text-slate-600 mb-1">Documentos (.pdf o .docx) — puedes elegir varios</label>
          <input id="file" type="file" accept=".pdf,.docx" multiple
                 class="block w-full text-xs text-slate-600 file:mr-3 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-medium file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100">
        </div>
        <div>
          <label class="block text-xs font-semibold text-slate-600 mb-1">Etiquetas (Enter o coma)</label>
          <div id="tagbox" class="flex flex-wrap items-center gap-1 border border-slate-300 rounded-md px-2 py-1.5">
            <input id="tagin" type="text" placeholder="Ej: Contrato, Ejecucion, Moquegua"
                   class="flex-1 min-w-[120px] outline-none text-xs py-0.5">
          </div>
        </div>
        <button id="btnAdd" onclick="agregarFuente()"
                class="w-full bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-md py-2 transition">
          + Agregar fuente
        </button>
        <p class="text-[11px] text-slate-400">Selecciona uno o varios archivos (Ctrl/Shift o arrástralos). El texto se procesa de forma temporal (OCR automático si el PDF está escaneado).</p>
        <p id="errAdd" class="hidden-x text-xs text-red-600 font-semibold bg-red-50 border border-red-200 rounded-md px-2 py-1.5"></p>
      </div>

      <div id="lista" class="flex-1 scroll-y px-4 py-3 space-y-2"></div>
      <div class="px-5 py-2 border-t border-slate-200 text-[11px] text-slate-400">
        <span id="contador">0</span> fuente(s) · <span id="activas">0</span> activa(s)
      </div>
    </div>
  </aside>

  <!-- ============ PANEL DERECHO (70%) : CHAT ============ -->
  <main class="flex-1 flex flex-col">
    <header class="px-6 py-4 bg-slate-900 text-white flex items-center justify-between gap-3">
      <div class="min-w-0">
        <h2 class="text-base font-semibold">Asistente de Contrataciones Públicas</h2>
        <p class="text-xs text-slate-300 truncate">Caso actual: <span id="casoEnChat">— (ninguno)</span></p>
      </div>
      <button id="btnLimpiar" onclick="limpiarChat()" title="Limpiar chat del caso actual"
              class="flex items-center gap-1.5 bg-slate-700 hover:bg-slate-600 text-white text-xs font-medium rounded-md px-3 py-2 transition whitespace-nowrap">
        <span>🧹</span> Limpiar Chat
      </button>
    </header>

    <div id="chat" class="flex-1 scroll-y px-6 py-5 space-y-4"></div>

    <div class="border-t border-slate-200 bg-white px-6 py-3">
      <div class="flex items-end gap-2">
        <textarea id="q" rows="1" placeholder="Entra a un caso para chatear..."
                  class="flex-1 resize-none border border-slate-300 rounded-lg px-3 py-2 text-sm outline-none focus:border-blue-500"
                  onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();enviar('chat');}"></textarea>
        <button id="btnChat" onclick="enviar('chat')"
                class="bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg px-4 py-2 transition">Enviar</button>
        <button id="btnAnal" onclick="enviar('analisis')"
                class="bg-amber-600 hover:bg-amber-700 text-white text-sm font-medium rounded-lg px-4 py-2 transition whitespace-nowrap">Ejecutar Análisis Legal</button>
      </div>
      <p class="text-[11px] text-slate-400 mt-1">"Enviar" = pregunta · "Ejecutar Análisis Legal" = auditoría de las fuentes activas guiada por sus etiquetas.</p>
    </div>
  </main>
</div>

<script>
let casoActual = null;   // {id, nombre}
let fuentes = [];        // fuentes del caso actual
let historial = [];      // memoria multi-turno del caso actual
const tags = [];
const MAX_HIST_CLIENTE = 40;

function esc(s){ return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function show(id){ document.getElementById(id).classList.remove('hidden-x'); }
function hide(id){ document.getElementById(id).classList.add('hidden-x'); }

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
    const r = await fetchConTimeout('/api/casos', {}, 20000);
    const casos = await r.json();
    if(!Array.isArray(casos) || !casos.length){
      cont.innerHTML = '<p class="text-xs text-slate-400 px-1 py-2">No hay casos todavía. Crea el primero arriba.</p>';
      return;
    }
    cont.innerHTML = '';
    casos.forEach(c=>{
      const card = document.createElement('div');
      card.className = 'border border-slate-200 rounded-lg p-3 hover:bg-blue-50/60 cursor-pointer flex items-start gap-2';
      card.onclick = ()=>entrarCaso(c);
      const fecha = (c.fecha_creacion||'').replace('T',' ').slice(0,16);
      card.innerHTML =
        '<div class="flex-1 min-w-0">' +
          '<div class="text-sm font-semibold text-slate-800 truncate">'+esc(c.nombre)+'</div>' +
          '<div class="text-[10px] text-slate-400 mt-0.5">'+(c.n_fuentes||0)+' fuente(s) · '+esc(fecha)+'</div>' +
        '</div>' +
        '<button title="Eliminar caso" class="text-slate-400 hover:text-red-600 text-sm leading-none">&times;</button>';
      card.querySelector('button').onclick = (ev)=>{ ev.stopPropagation(); borrarCaso(c); };
      cont.appendChild(card);
    });
  }catch(e){
    cont.innerHTML = '<p class="text-xs text-red-600 px-1 py-2">Error al cargar casos: '+esc(String(e))+'</p>';
  }
}
async function crearCaso(){
  const inp = document.getElementById('nombreCaso');
  const nombre = inp.value.trim();
  if(!nombre){ inp.focus(); return; }
  try{
    const r = await fetchConTimeout('/api/casos', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({nombre})}, 20000);
    const c = await r.json();
    inp.value='';
    entrarCaso(c);
  }catch(e){ alert('No se pudo crear el caso: '+e); }
}
async function borrarCaso(c){
  if(!confirm('¿Eliminar el caso "'+c.nombre+'" y todos sus documentos?')) return;
  await fetch('/api/casos/'+c.id, {method:'DELETE'});
  cargarCasos();
}

// =================== ENTRAR / SALIR DE UN CASO ===================
function setChat(enabled){
  ['q','btnChat','btnAnal','btnLimpiar'].forEach(id=>document.getElementById(id).disabled = !enabled);
  document.getElementById('q').placeholder = enabled ? 'Escribe tu consulta...' : 'Entra a un caso para chatear...';
}
function resetChatUI(msg){
  historial = [];
  document.getElementById('chat').innerHTML =
    '<div class="flex"><div class="bg-white border border-slate-200 rounded-2xl px-4 py-3 max-w-[85%] shadow-sm prosa text-sm">'+msg+'</div></div>';
}
async function entrarCaso(c){
  casoActual = c;
  fuentes = []; tags.length = 0; pintarTags(); mostrarErrorAdd('');
  document.getElementById('tituloCaso').textContent = c.nombre;
  document.getElementById('casoEnChat').textContent = c.nombre;
  hide('vistaCasos'); show('vistaFuentes');
  setChat(true);
  resetChatUI('Estás en el caso <b>'+esc(c.nombre)+'</b>. Sube documentos, actívalos y pregúntame. El chat y las fuentes son exclusivos de este caso.');
  // Cargar fuentes del caso (aislado).
  try{
    const r = await fetch('/api/casos/'+c.id+'/fuentes');
    const d = await r.json();
    fuentes = (Array.isArray(d)?d:[]).map(f=>({...f, activo:true}));
    pintarFuentes();
    fuentes.forEach(f=>{ if((f.estado||'listo')==='procesando') pollFuente(f.id); });
  }catch(e){ document.getElementById('lista').innerHTML = '<p class="text-xs text-red-600">Error al cargar fuentes.</p>'; }
}
function volverCasos(){
  casoActual = null; fuentes = []; historial = [];
  hide('vistaFuentes'); show('vistaCasos');
  setChat(false);
  document.getElementById('casoEnChat').textContent = '— (ninguno)';
  resetChatUI('Selecciona un caso en el panel izquierdo o crea uno nuevo para empezar.');
  cargarCasos();
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
  if(!fuentes.length){ cont.innerHTML = '<p class="text-xs text-slate-400 px-1 py-2">Aún no hay fuentes en este caso.</p>'; contar(); return; }
  cont.innerHTML = '';
  fuentes.forEach(f=>{
    const estado = f.estado || 'listo';
    const listo = estado === 'listo';
    const card = document.createElement('div');
    card.className = 'border rounded-lg p-3 ' +
      (estado==='error' ? 'bg-red-50 border-red-200'
        : (listo && f.activo) ? 'bg-blue-50/60 border-slate-200'
        : 'bg-slate-50 border-slate-200');
    const tagsHtml = (f.etiquetas||[]).map(t=>'<span class="bg-slate-200 text-slate-600 rounded px-1.5 py-0.5 text-[10px]">'+esc(t)+'</span>').join(' ');
    const dis = listo ? '' : 'disabled';
    const toggle = '<label class="switch mt-0.5"><input type="checkbox" '+(f.activo&&listo?'checked':'')+' '+dis+
                   ' onchange="toggleFuente(\''+f.id+'\', this.checked)"><span class="slider"></span></label>';
    let estadoHtml;
    if(estado==='procesando'){
      estadoHtml = '<div class="text-[10px] text-amber-600 font-semibold mb-1 flex items-center gap-1">' +
                   '<span class="inline-block w-3 h-3 border-2 border-amber-500 border-t-transparent rounded-full animate-spin"></span>' +
                   'Procesando (extracción / OCR)...</div>';
    } else if(estado==='error'){
      estadoHtml = '<div class="text-[10px] text-red-600 font-semibold mb-1">Error: '+esc(f.error||'no se pudo procesar')+'</div>';
    } else {
      const ocr = f.metodo==='ocr' ? '<span class="text-[10px] text-green-700 font-semibold">OCR</span>' : '';
      estadoHtml = '<div class="text-[10px] text-slate-400 mb-1">'+(f.n_chars||0).toLocaleString()+' chars '+ocr+'</div>';
    }
    card.innerHTML =
      '<div class="flex items-start gap-2">' + toggle +
        '<div class="flex-1 min-w-0">' +
          '<div class="text-xs font-semibold text-slate-800 truncate" title="'+esc(f.nombre)+'">'+esc(f.nombre)+'</div>' +
          estadoHtml +
          '<div class="flex flex-wrap gap-1">'+tagsHtml+'</div>' +
        '</div>' +
        '<button onclick="borrarFuente(\''+f.id+'\')" class="text-slate-400 hover:text-red-600 text-sm leading-none">&times;</button>' +
      '</div>';
    cont.appendChild(card);
  });
  contar();
}
function toggleFuente(id, val){ const f=fuentes.find(x=>x.id===id); if(f){ f.activo=val; } contar(); pintarFuentes(); }

function pollFuente(id){
  const casoId = casoActual ? casoActual.id : null;
  let intentos = 0;
  const iv = setInterval(async ()=>{
    intentos++;
    if(!casoActual || casoActual.id !== casoId){ clearInterval(iv); return; }  // salio del caso
    const f = fuentes.find(x=>x.id===id);
    if(!f){ clearInterval(iv); return; }
    if(intentos > 120){ clearInterval(iv); f.estado='error'; f.error='Tiempo de espera agotado (5 min).'; pintarFuentes(); return; }
    try{
      const r = await fetch('/api/casos/'+casoId+'/fuentes/'+id);
      if(r.status===404){ clearInterval(iv); return; }
      if(!r.ok) return;
      const d = await r.json();
      f.estado=d.estado; f.n_chars=d.n_chars; f.metodo=d.metodo; f.error=d.error;
      if(d.estado!=='procesando'){ clearInterval(iv); }
      pintarFuentes();
    }catch(e){ /* reintenta */ }
  }, 2500);
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
  if(!archivos.length){ mostrarErrorAdd('Selecciona uno o más archivos .pdf o .docx.'); return; }

  // Las etiquetas actuales se aplican a todo el lote.
  const etiquetasLote = tags.slice();
  const casoId = casoActual.id;

  // LIMPIEZA DEL FORMULARIO: listo de inmediato para una nueva subida.
  input.value = '';
  tags.length = 0; pintarTags();

  // Dispara cada subida de forma concurrente (no se espera una por una).
  archivos.forEach(f => subirUnArchivo(casoId, f, etiquetasLote));
}

// Sube UN archivo: crea su tarjeta 'Procesando...' y arranca su propio polling.
async function subirUnArchivo(casoId, file, etiquetasArr){
  const fd = new FormData();
  fd.append('archivo', file);
  fd.append('etiquetas', etiquetasArr.join(','));
  try{
    const r = await fetchConTimeout('/api/casos/'+casoId+'/fuentes', {method:'POST', body:fd}, 60000);
    let d = {}; try { d = await r.json(); } catch(_){ d = {}; }
    // Si el usuario salio del caso mientras subia, descartar el resultado.
    if(!casoActual || casoActual.id !== casoId) return;
    if(!r.ok){ mostrarErrorAdd('Error '+r.status+' al subir "'+file.name+'": '+(d.error||'fallo al subir.')); return; }
    if(d.error){ mostrarErrorAdd('"'+file.name+'": '+d.error); return; }
    d.activo = true; fuentes.push(d); pintarFuentes();
    if((d.estado||'listo')==='procesando'){ pollFuente(d.id); }
  }catch(e){
    if(!casoActual || casoActual.id !== casoId) return;
    if(e && e.name==='AbortError'){ mostrarErrorAdd('La subida de "'+file.name+'" tardó demasiado y se canceló (timeout).'); }
    else { mostrarErrorAdd('No se pudo subir "'+file.name+'": ' + (e && e.message ? e.message : e)); }
  }
}

// =================== CHAT ===================
const chat = document.getElementById('chat');
function limpiarChat(){
  if(!casoActual) return;
  resetChatUI('Chat reiniciado para el caso <b>'+esc(casoActual.nombre)+'</b>.');
}
function addMsg(html, lado){
  const wrap = document.createElement('div'); wrap.className = 'flex ' + (lado==='user'?'justify-end':'');
  const b = document.createElement('div');
  b.className = (lado==='user'
    ? 'bg-blue-600 text-white rounded-2xl px-4 py-3 max-w-[85%] shadow-sm prosa text-sm'
    : 'bg-white border border-slate-200 rounded-2xl px-4 py-3 max-w-[85%] shadow-sm prosa text-sm');
  b.innerHTML = html; wrap.appendChild(b); chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight; return b;
}
async function enviar(modo){
  if(!casoActual){ return; }
  const q = document.getElementById('q');
  const texto = q.value.trim();
  const activas = fuentes.filter(f=>f.activo && (f.estado||'listo')==='listo').map(f=>f.id);
  if(modo==='chat' && !texto && !activas.length){ return; }
  if(modo==='analisis' && !activas.length){ addMsg('Activa al menos una fuente (lista) para ejecutar el análisis legal.', 'bot'); return; }

  const userTxt = texto || (modo==='analisis' ? '[Solicitud de análisis legal de las fuentes activas]' : '');
  if(texto) addMsg(esc(texto), 'user');
  else if(modo==='analisis') addMsg('<i>Ejecutar análisis legal de las fuentes activas</i>', 'user');
  q.value='';
  document.getElementById('btnChat').disabled=true; document.getElementById('btnAnal').disabled=true;
  const cargando = addMsg('<span class="text-slate-400 italic">Analizando con las fuentes activas y la normativa...</span>', 'bot');
  try{
    const r = await fetchConTimeout('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({caso_id: casoActual.id, pregunta: texto, fuentes_activas: activas, modo: modo, k: 5,
                            historial: historial.slice(-MAX_HIST_CLIENTE)})}, 180000);
    let d = {}; try { d = await r.json(); } catch(_){ d = {}; }
    if(!r.ok){ cargando.innerHTML = '<span class="text-amber-700">Error '+r.status+': '+esc(d.error||'fallo del servidor')+'</span>'; return; }
    if(d.error){ cargando.innerHTML = '<span class="text-amber-700">'+esc(d.error)+'</span>'; return; }
    let h = esc(d.respuesta || 'Sin respuesta.');
    if(d.fuentes_usadas && d.fuentes_usadas.length){
      h += '<div class="mt-2 pt-2 border-t border-slate-100 text-[11px] text-slate-500"><b>Fuentes del caso usadas:</b> '
         + d.fuentes_usadas.map(x=>esc(x.nombre)).join(', ') + '</div>';
    }
    if(d.fuentes_normativas && d.fuentes_normativas.length){
      h += '<div class="mt-1 text-[11px] text-slate-500"><b>Normas cruzadas:</b><br>'
         + d.fuentes_normativas.map(s=>'<span class="inline-block bg-slate-100 border border-slate-200 rounded px-1.5 py-0.5 mt-1 mr-1">'+esc(s.documento)+', Art. '+esc(String(s.articulo_num))+'</span>').join('') + '</div>';
    }
    cargando.innerHTML = h;
    if(userTxt) historial.push({rol:'user', texto:userTxt});
    historial.push({rol:'model', texto: d.respuesta || ''});
    if(historial.length > MAX_HIST_CLIENTE) historial = historial.slice(-MAX_HIST_CLIENTE);
  }catch(e){
    if(e && e.name==='AbortError'){ cargando.innerHTML = '<span class="text-amber-700">La consulta tardó demasiado y se canceló (timeout).</span>'; }
    else { cargando.innerHTML = '<span class="text-amber-700">Error: '+esc(String(e))+'</span>'; }
  }
  finally{ document.getElementById('btnChat').disabled=false; document.getElementById('btnAnal').disabled=false; chat.scrollTop=chat.scrollHeight; }
}

// =================== INIT ===================
setChat(false);
resetChatUI('Selecciona un caso en el panel izquierdo o crea uno nuevo para empezar.');
cargarCasos();
</script>
</body>
</html>
"""
