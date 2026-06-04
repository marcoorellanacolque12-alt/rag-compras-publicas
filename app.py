"""
app.py
-------------------------------------------------------------------
Asistente RAG de Contrataciones Publicas — interfaz tipo NotebookLM
("Fuentes Seleccionables").

Layout dividido:
  - Panel izquierdo (30%): gestion de "Fuentes del Caso" (subir documentos
    con OCR + etiquetas; cada fuente tiene un interruptor on/off).
  - Panel derecho (70%): chat y resultados del analisis.

Logica de inyeccion de contexto:
  - Las fuentes subidas se guardan en el servidor con un id.
  - En cada turno, el frontend envia SOLO los ids de las fuentes ENCENDIDAS.
  - El backend concatena unicamente esos textos como contexto, cruzandolos
    con la base vectorial (BigQuery VECTOR_SEARCH). Las fuentes apagadas se
    ignoran en ese turno.

Endpoints:
  GET    /                 -> interfaz web (NotebookLM-like).
  POST   /api/fuentes      -> multipart {archivo, etiquetas} -> registra fuente.
  GET    /api/fuentes      -> lista las fuentes del caso.
  DELETE /api/fuentes/{id} -> elimina una fuente.
  POST   /api/chat         -> {pregunta, fuentes_activas[], modo, k} -> respuesta.
  GET    /salud            -> healthcheck.

Nota: el estado de fuentes vive EN MEMORIA del proceso (suficiente para un
servidor uvicorn unico). Para Cloud Run multi-instancia conviene moverlo a
Firestore/Redis.

Requisitos:
    pip install fastapi "uvicorn[standard]" python-multipart google-cloud-bigquery \
                google-genai pypdf python-docx pymupdf google-cloud-vision
    Autenticacion ADC + tabla creada con indexar_bigquery.py + Vision API.

Ejecutar en local:
    python -m uvicorn app:app --host 0.0.0.0 --port 8080
-------------------------------------------------------------------
"""

import uuid
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from google.genai.types import GenerateContentConfig, Content, Part

# Motor RAG ya implementado y probado.
from responder import recuperar, construir_contexto, cliente, MODELO_GEN
# Extraccion de texto reutilizada (en memoria) con FALLBACK de OCR (Cloud Vision).
from extraccion_texto import extraer_pdf_inteligente, extraer_docx

app = FastAPI(title="Asistente RAG - Contrataciones Publicas (NotebookLM)")

# Estado de las "Fuentes del Caso" (en memoria del proceso).
#   id -> {id, nombre, etiquetas[], texto, n_chars, metodo}
FUENTES = {}

# Limite total de texto de fuentes activas que se envia al LLM (control de costo).
MAX_CONTEXT_CHARS = 200_000
# Caracteres de las fuentes usados para construir la consulta de recuperacion.
QUERY_DOC_CHARS = 3_000


class Turno(BaseModel):
    rol: str = "user"           # "user" | "model"
    texto: str = ""


class Mensaje(BaseModel):
    pregunta: str = ""
    fuentes_activas: list[str] = []
    modo: str = "chat"          # "chat" | "analisis"
    k: int = 5
    historial: list[Turno] = []  # turnos previos (memoria multi-turno)


# Maximo de turnos de historial que se inyectan al modelo (control de tokens/costo).
MAX_TURNOS_HISTORIAL = 12


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


def _sistema_chat():
    return (
        "Eres un asistente experto en contrataciones publicas. Respondes consultas "
        "de areas usuarias, especialistas y operadores. Reglas:\n"
        "1. Usa como contexto los DOCUMENTOS DEL CASO (fuentes activas) y las NORMAS "
        "RECUPERADAS del marco legal. No uses conocimiento externo.\n"
        "2. Si la respuesta no esta en el contexto, dilo claramente: no inventes.\n"
        "3. Cita SIEMPRE el respaldo: articulos de la norma (ej: Reglamento, Art. 304) "
        "y/o el nombre de la fuente del caso.\n"
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


# ============================ SALUD ============================
@app.get("/salud")
def salud():
    return {"status": "ok", "fuentes_en_memoria": len(FUENTES)}


# ====================== GESTION DE FUENTES =====================
def _fuente_publica(f):
    """Metadatos de una fuente (sin el texto completo)."""
    return {
        "id": f["id"], "nombre": f["nombre"], "etiquetas": f["etiquetas"],
        "n_chars": f["n_chars"], "metodo": f["metodo"],
        "estado": f["estado"], "error": f.get("error"),
    }


def _procesar_fuente(fid, nombre, data):
    """
    Worker en SEGUNDO PLANO (threadpool): extrae el texto (con OCR si aplica)
    y actualiza el estado de la fuente a 'listo' o 'error'. Para PDFs escaneados
    grandes esto puede tardar, pero la subida ya respondio al usuario.
    """
    f = FUENTES.get(fid)
    if f is None:
        return  # la fuente fue eliminada antes de terminar
    try:
        texto, metodo = _extraer_texto_temporal(nombre, data)
        if not texto or len(texto.strip()) < 20:
            f.update(estado="error",
                     error="No se pudo extraer texto del documento ni con OCR (archivo vacio o danado).")
            return
        f.update(texto=texto, n_chars=len(texto), metodo=metodo, estado="listo", error=None)
    except ValueError as e:
        f.update(estado="error", error=str(e))
    except Exception as e:
        f.update(estado="error", error=f"{type(e).__name__}: {e}")


@app.post("/api/fuentes")
async def subir_fuente(background_tasks: BackgroundTasks,
                       archivo: UploadFile = File(...), etiquetas: str = Form("")):
    # SUBIDA ASINCRONA CON ESTADO: responde de inmediato con estado="procesando"
    # y el OCR/extraccion (potencialmente lento) corre en segundo plano.
    try:
        # Etiquetas robustas: campo vacio, ausente o solo comas -> lista vacia.
        lista_etiquetas = [e.strip() for e in (etiquetas or "").split(",") if e.strip()]

        # Validacion de formato temprana (para fallar rapido, antes de registrar).
        lower = (archivo.filename or "").lower()
        if not (lower.endswith(".pdf") or lower.endswith(".docx")):
            return JSONResponse(status_code=400, content={
                "error": "Formato no soportado. Solo se admiten .pdf o .docx."})

        data = await archivo.read()
        if not data:
            return JSONResponse(status_code=400, content={"error": "El archivo llego vacio."})

        # Registrar la fuente en estado 'procesando' y devolver de inmediato.
        fid = uuid.uuid4().hex[:12]
        FUENTES[fid] = {
            "id": fid, "nombre": archivo.filename, "etiquetas": lista_etiquetas,
            "texto": "", "n_chars": 0, "metodo": None,
            "estado": "procesando", "error": None,
        }
        # Procesamiento pesado en segundo plano (threadpool, tras enviar la respuesta).
        background_tasks.add_task(_procesar_fuente, fid, archivo.filename, data)

        return JSONResponse(status_code=202, content=_fuente_publica(FUENTES[fid]))

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": f"Error interno al registrar el documento: {type(e).__name__}: {e}"
        })


@app.get("/api/fuentes/{fid}")
def estado_fuente(fid: str):
    """Estado de una fuente (para polling del frontend)."""
    f = FUENTES.get(fid)
    if f is None:
        return JSONResponse(status_code=404, content={"error": "Fuente no encontrada."})
    return _fuente_publica(f)


@app.get("/api/fuentes")
def listar_fuentes():
    return [_fuente_publica(f) for f in FUENTES.values()]


@app.delete("/api/fuentes/{fid}")
def borrar_fuente(fid: str):
    FUENTES.pop(fid, None)
    return {"ok": True, "restantes": len(FUENTES)}


# =========================== CHAT/ANALISIS =====================
@app.post("/api/chat")
def chat(m: Mensaje):
    pregunta = (m.pregunta or "").strip()

    # Fuentes activas: solo las encendidas, que existan y que esten LISTAS
    # (las que aun estan 'procesando' o en 'error' se ignoran en este turno).
    activos = [FUENTES[i] for i in m.fuentes_activas
               if i in FUENTES and FUENTES[i].get("estado") == "listo"]
    etiquetas = sorted({e for f in activos for e in f["etiquetas"]})

    if not pregunta and not activos:
        return JSONResponse(status_code=400, content={
            "error": "Escribe una consulta o activa al menos una fuente del caso."
        })

    # Contexto de las fuentes activas (concatenado y acotado).
    partes = []
    for f in activos:
        cab = f"[FUENTE: {f['nombre']} | etiquetas: {', '.join(f['etiquetas']) or '-'}]"
        partes.append(cab + "\n" + f["texto"])
    contexto_fuentes = "\n\n".join(partes)[:MAX_CONTEXT_CHARS]

    # Recuperacion en la base vectorial (guiada por la pregunta + etiquetas + extracto).
    base_query = pregunta or (", ".join(etiquetas))
    consulta = (base_query + "\n" + contexto_fuentes[:QUERY_DOC_CHARS]).strip()
    filas = recuperar(consulta, k=m.k) if consulta else []
    contexto_normas = construir_contexto(filas) if filas else "(sin normas recuperadas)"

    # Construccion del prompt segun el modo.
    secciones = [f"NORMAS RECUPERADAS (base vectorial):\n{contexto_normas}"]
    if contexto_fuentes:
        secciones.append("DOCUMENTOS DEL CASO (fuentes activas seleccionadas por el usuario):\n"
                         + contexto_fuentes)
    else:
        secciones.append("DOCUMENTOS DEL CASO: (ninguna fuente activa en este turno).")

    if m.modo == "analisis":
        sistema = _sistema_auditoria(etiquetas)
        secciones.append("TAREA: Realiza la auditoria legal de las fuentes activas segun las "
                         "instrucciones del sistema." + (f"\nFoco adicional del usuario: {pregunta}" if pregunta else ""))
    else:
        sistema = _sistema_chat()
        secciones.append(f"CONSULTA DEL USUARIO:\n{pregunta or '(resume y comenta las fuentes activas)'}")

    prompt = "\n\n".join(secciones)

    # MEMORIA MULTI-TURNO: el historial previo se inyecta como turnos nativos
    # user/model; la nueva pregunta (con el contexto RAG + documentos activos)
    # va como turno final 'user'. Las instrucciones del sistema van en config.
    contents = []
    for t in m.historial[-MAX_TURNOS_HISTORIAL:]:
        rol = "model" if (t.rol or "").lower() in ("model", "assistant", "ia", "bot") else "user"
        txt = (t.texto or "").strip()
        if txt:
            contents.append(Content(role=rol, parts=[Part(text=txt)]))
    # Gemini exige que la conversacion empiece con un turno 'user'.
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
<title>Asistente de Contrataciones Publicas</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  .scroll-y { overflow-y: auto; }
  /* Interruptor (toggle) */
  .switch { position: relative; display: inline-block; width: 38px; height: 22px; flex: none; }
  .switch input { opacity: 0; width: 0; height: 0; }
  .slider { position: absolute; cursor: pointer; inset: 0; background: #cbd5e1; border-radius: 9999px; transition: .2s; }
  .slider:before { content: ""; position: absolute; height: 16px; width: 16px; left: 3px; top: 3px; background: #fff; border-radius: 9999px; transition: .2s; }
  input:checked + .slider { background: #2563eb; }
  input:checked + .slider:before { transform: translateX(16px); }
  .prosa { white-space: pre-wrap; line-height: 1.6; }
</style>
</head>
<body class="h-screen bg-slate-100 text-slate-800">
<div class="flex h-screen">

  <!-- ============ PANEL IZQUIERDO (30%) : FUENTES ============ -->
  <aside class="w-[30%] min-w-[300px] max-w-[460px] bg-white border-r border-slate-200 flex flex-col">
    <div class="px-5 py-4 border-b border-slate-200">
      <h1 class="text-base font-semibold text-slate-900">Fuentes del Caso</h1>
      <p class="text-xs text-slate-500 mt-1">Sube documentos y actívalos con el interruptor para usarlos como contexto.</p>
    </div>

    <!-- Subida -->
    <div class="px-5 py-4 border-b border-slate-200 space-y-3">
      <div>
        <label class="block text-xs font-semibold text-slate-600 mb-1">Documento (.pdf o .docx)</label>
        <input id="file" type="file" accept=".pdf,.docx"
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
      <p class="text-[11px] text-slate-400">El texto se procesa de forma temporal (OCR automático si el PDF está escaneado). No se almacena en la base vectorial.</p>
      <p id="errAdd" class="hidden text-xs text-red-600 font-semibold bg-red-50 border border-red-200 rounded-md px-2 py-1.5"></p>
    </div>

    <!-- Lista de fuentes -->
    <div id="lista" class="flex-1 scroll-y px-4 py-3 space-y-2"></div>

    <div class="px-5 py-2 border-t border-slate-200 text-[11px] text-slate-400">
      <span id="contador">0</span> fuente(s) · <span id="activas">0</span> activa(s)
    </div>
  </aside>

  <!-- ============ PANEL DERECHO (70%) : CHAT ============ -->
  <main class="flex-1 flex flex-col">
    <header class="px-6 py-4 bg-slate-900 text-white flex items-center justify-between gap-3">
      <div>
        <h2 class="text-base font-semibold">Asistente de Contrataciones Públicas</h2>
        <p class="text-xs text-slate-300">El modelo responde usando solo las fuentes activas + el marco normativo (Ley y Reglamento).</p>
      </div>
      <button id="btnLimpiar" onclick="limpiarChat()" title="Limpiar chat y reiniciar la memoria de la conversación"
              class="flex items-center gap-1.5 bg-slate-700 hover:bg-slate-600 text-white text-xs font-medium rounded-md px-3 py-2 transition whitespace-nowrap">
        <span>🧹</span> Limpiar Chat
      </button>
    </header>

    <div id="chat" class="flex-1 scroll-y px-6 py-5 space-y-4">
      <div class="flex">
        <div class="bg-white border border-slate-200 rounded-2xl px-4 py-3 max-w-[85%] shadow-sm prosa text-sm">
Hola. Sube documentos en el panel izquierdo, actívalos con su interruptor y pregúntame. También puedo cruzar tus documentos con la normativa. Si no activas ninguna fuente, respondo solo con base en la Ley y el Reglamento.
        </div>
      </div>
    </div>

    <div class="border-t border-slate-200 bg-white px-6 py-3">
      <div class="flex items-end gap-2">
        <textarea id="q" rows="1" placeholder="Escribe tu consulta..."
                  class="flex-1 resize-none border border-slate-300 rounded-lg px-3 py-2 text-sm outline-none focus:border-blue-500"
                  onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();enviar('chat');}"></textarea>
        <button id="btnChat" onclick="enviar('chat')"
                class="bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg px-4 py-2 transition">Enviar</button>
        <button id="btnAnal" onclick="enviar('analisis')"
                class="bg-amber-600 hover:bg-amber-700 text-white text-sm font-medium rounded-lg px-4 py-2 transition whitespace-nowrap">Ejecutar Análisis Legal</button>
      </div>
      <p class="text-[11px] text-slate-400 mt-1">"Enviar" = pregunta normal · "Ejecutar Análisis Legal" = auditoría de las fuentes activas guiada por sus etiquetas.</p>
    </div>
  </main>
</div>

<script>
let fuentes = [];     // {id, nombre, etiquetas[], metodo, activo}
const tags = [];      // etiquetas en edicion
let historial = [];   // memoria multi-turno: [{rol:'user'|'model', texto}]
const MAX_HIST_CLIENTE = 40;  // tope para no crecer sin limite en el navegador

function esc(s){ return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

// fetch con timeout via AbortController: si el servidor demora demasiado, aborta.
async function fetchConTimeout(url, opts, ms){
  const ctrl = new AbortController();
  const t = setTimeout(()=>ctrl.abort(), ms);
  try { return await fetch(url, {...opts, signal: ctrl.signal}); }
  finally { clearTimeout(t); }
}
// Muestra/oculta el mensaje de error del panel de subida.
function mostrarErrorAdd(msg){
  const el = document.getElementById('errAdd');
  if(!msg){ el.classList.add('hidden'); el.textContent=''; return; }
  el.textContent = msg; el.classList.remove('hidden');
}

// ---------- Etiquetas (chips) ----------
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

// ---------- Fuentes ----------
function contar(){
  document.getElementById('contador').textContent = fuentes.length;
  document.getElementById('activas').textContent = fuentes.filter(f=>f.activo && (f.estado||'listo')==='listo').length;
}
function pintarFuentes(){
  const cont = document.getElementById('lista');
  if(!fuentes.length){ cont.innerHTML = '<p class="text-xs text-slate-400 px-1 py-2">Aún no hay fuentes. Agrega un documento arriba.</p>'; contar(); return; }
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

    // Interruptor: deshabilitado mientras la fuente no este 'listo'.
    const dis = listo ? '' : 'disabled';
    const toggle = '<label class="switch mt-0.5"><input type="checkbox" '+(f.activo&&listo?'checked':'')+' '+dis+
                   ' onchange="toggleFuente(\''+f.id+'\', this.checked)"><span class="slider"></span></label>';

    // Linea de estado segun procesando / listo / error.
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
      '<div class="flex items-start gap-2">' +
        toggle +
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

// Polling del estado de una fuente hasta 'listo' o 'error'.
function pollFuente(id){
  let intentos = 0;
  const iv = setInterval(async ()=>{
    intentos++;
    const f = fuentes.find(x=>x.id===id);
    if(!f){ clearInterval(iv); return; }              // fue eliminada
    if(intentos > 120){ clearInterval(iv); f.estado='error'; f.error='Tiempo de espera agotado (5 min).'; pintarFuentes(); return; }
    try{
      const r = await fetch('/api/fuentes/'+id);
      if(r.status===404){ clearInterval(iv); return; }
      if(!r.ok) return;                                // reintenta en el proximo tick
      const d = await r.json();
      f.estado=d.estado; f.n_chars=d.n_chars; f.metodo=d.metodo; f.error=d.error;
      if(d.estado!=='procesando'){ clearInterval(iv); }
      pintarFuentes();
    }catch(e){ /* error transitorio: reintenta */ }
  }, 2500);
}
async function borrarFuente(id){
  await fetch('/api/fuentes/'+id, {method:'DELETE'});
  fuentes = fuentes.filter(x=>x.id!==id); pintarFuentes();
}
async function agregarFuente(){
  mostrarErrorAdd('');  // limpia errores previos
  if(tagin.value){ agregarTag(tagin.value); tagin.value=''; }
  const f = document.getElementById('file').files[0];
  if(!f){ mostrarErrorAdd('Selecciona un archivo .pdf o .docx.'); return; }

  const btn = document.getElementById('btnAdd');
  btn.disabled = true; btn.textContent = 'Procesando...';
  const fd = new FormData(); fd.append('archivo', f); fd.append('etiquetas', tags.join(','));

  // CORRECCION 2: try/catch robusto con timeout. Pase lo que pase (timeout,
  // error 500, red caida), el boton SIEMPRE vuelve a su estado y se muestra el error.
  try{
    const r = await fetchConTimeout('/api/fuentes', {method:'POST', body:fd}, 180000); // 3 min
    let d = {};
    try { d = await r.json(); } catch(_){ d = {}; }

    if(!r.ok){
      mostrarErrorAdd('El servidor respondió con error ' + r.status + ': ' + (d.error || 'fallo interno al procesar el documento.'));
      return;
    }
    if(d.error){ mostrarErrorAdd(d.error); return; }

    // La fuente queda 'procesando'; se activa por defecto y se hace polling hasta 'listo'.
    d.activo = true; fuentes.push(d); pintarFuentes();
    if((d.estado||'listo') === 'procesando'){ pollFuente(d.id); }
    document.getElementById('file').value=''; tags.length=0; pintarTags();
  }catch(e){
    if(e && e.name === 'AbortError'){
      mostrarErrorAdd('La subida tardó demasiado y se canceló (timeout). Intenta con un archivo más liviano o reinténtalo.');
    } else {
      mostrarErrorAdd('No se pudo subir el documento: ' + (e && e.message ? e.message : e));
    }
  }finally{
    // GARANTIZADO: el boton sale de "Procesando..." en todos los casos.
    btn.disabled = false; btn.textContent = '+ Agregar fuente';
  }
}

// ---------- Chat ----------
const chat = document.getElementById('chat');
const BIENVENIDA = '<div class="flex"><div class="bg-white border border-slate-200 rounded-2xl px-4 py-3 max-w-[85%] shadow-sm prosa text-sm">Hola. Sube documentos en el panel izquierdo, actívalos con su interruptor y pregúntame. Puedo recordar el hilo de la conversación para preguntas de seguimiento; usa "🧹 Limpiar Chat" para reiniciar la memoria al cambiar de tema.</div></div>';

// Limpia el chat y REINICIA la memoria multi-turno.
function limpiarChat(){
  historial = [];
  chat.innerHTML = BIENVENIDA;
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
  const q = document.getElementById('q');
  const texto = q.value.trim();
  // Solo fuentes activas Y listas (las que aun procesan o fallaron no cuentan).
  const activas = fuentes.filter(f=>f.activo && (f.estado||'listo')==='listo').map(f=>f.id);
  if(modo==='chat' && !texto && !activas.length){ return; }
  if(modo==='analisis' && !activas.length){ addMsg('Activa al menos una fuente (lista) para ejecutar el análisis legal.', 'bot'); return; }

  // Texto que representa este turno del usuario en la memoria.
  const userTxt = texto || (modo==='analisis' ? '[Solicitud de análisis legal de las fuentes activas]' : '');
  if(texto) addMsg(esc(texto), 'user');
  else if(modo==='analisis') addMsg('<i>Ejecutar análisis legal de las fuentes activas</i>', 'user');
  q.value='';
  document.getElementById('btnChat').disabled=true; document.getElementById('btnAnal').disabled=true;
  const cargando = addMsg('<span class="text-slate-400 italic">Analizando con las fuentes activas y la normativa...</span>', 'bot');
  try{
    const r = await fetchConTimeout('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({pregunta: texto, fuentes_activas: activas, modo: modo, k: 5,
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

    // Registrar el turno en la memoria multi-turno (texto plano).
    if(userTxt) historial.push({rol:'user', texto:userTxt});
    historial.push({rol:'model', texto: d.respuesta || ''});
    if(historial.length > MAX_HIST_CLIENTE) historial = historial.slice(-MAX_HIST_CLIENTE);
  }catch(e){
    if(e && e.name==='AbortError'){ cargando.innerHTML = '<span class="text-amber-700">La consulta tardó demasiado y se canceló (timeout).</span>'; }
    else { cargando.innerHTML = '<span class="text-amber-700">Error: '+esc(String(e))+'</span>'; }
  }
  finally{ document.getElementById('btnChat').disabled=false; document.getElementById('btnAnal').disabled=false; chat.scrollTop=chat.scrollHeight; }
}

// ---------- Init ----------
(async function(){
  try{ const r = await fetch('/api/fuentes'); const d = await r.json();
    fuentes = (d||[]).map(f=>({...f, activo:true})); }catch(e){}
  pintarFuentes();
  // Reanudar polling de cualquier fuente que siga procesando (p.ej. tras recargar).
  fuentes.forEach(f=>{ if((f.estado||'listo')==='procesando') pollFuente(f.id); });
})();
</script>
</body>
</html>
"""

