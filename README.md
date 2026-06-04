# Asistente RAG de Contrataciones Públicas

Sistema RAG (Retrieval-Augmented Generation) para consultar y auditar el marco
normativo de contrataciones públicas (Ley N° 32069 y su Reglamento), pensado para
áreas usuarias, especialistas y operadores. Construido sobre Google Cloud.

## Arquitectura

```
PDFs/DOCX ─► extracción de texto (+ OCR) ─► chunking por artículo ─► embeddings ─► BigQuery (VECTOR_SEARCH)
                                                                                          │
Usuario ─► interfaz web (NotebookLM) ─► /api/chat ─► recuperación (RAG) + Gemini ─────────┘
```

Todo el stack es **sin costo fijo**: almacenamiento en GCS + BigQuery serverless
(`VECTOR_SEARCH`), embeddings y Gemini bajo demanda. El despliegue web (Cloud Run)
escala a cero.

## Pipeline de datos (ejecutar en orden)

| Script | Función |
|--------|---------|
| `crear_datalake.py` | Crea el bucket de GCS y la estructura de carpetas normativas. |
| `subida_masiva.py` | Carga masiva de `.pdf`/`.docx` (con OCR opcional vía `--habilitar-ocr-masivo`). |
| `extraccion_texto.py` | Extrae texto plano; fallback de OCR (Cloud Vision) para escaneos. |
| `chunking_articulos.py` | Trocea los textos por artículo (IDs únicos). |
| `generar_embeddings.py` | Genera embeddings (`text-multilingual-embedding-002`, 768 dims). |
| `indexar_bigquery.py` | Carga chunks + vectores en BigQuery. |
| `buscar.py` | Búsqueda semántica por consola (solo recuperación). |
| `responder.py` | RAG completo por consola (recuperación + generación con citas). |

## Aplicación web

| Archivo | Función |
|--------|---------|
| `app.py` | API + interfaz web tipo NotebookLM (FastAPI + Tailwind). |
| `requirements.txt`, `Dockerfile`, `.dockerignore` | Empaquetado para contenedor. |
| `desplegar_cloudrun.ps1` | Despliegue a Google Cloud Run (IAM + APIs + deploy). |

### Características de la web
- **Fuentes Seleccionables:** panel lateral para subir documentos (con OCR automático
  para PDFs escaneados) y activarlos/desactivarlos con un interruptor. Solo las fuentes
  encendidas se inyectan como contexto.
- **Subida asíncrona con estado:** `procesando → listo/error` con polling.
- **Análisis legal guiado por etiquetas:** auditoría de documentos cruzada con la normativa.
- **Memoria multi-turno:** historial de conversación (roles nativos `user`/`model`),
  con botón para limpiar el chat.

### Ejecutar en local
```powershell
python -m uvicorn app:app --host 0.0.0.0 --port 8080
# abrir http://localhost:8080
```

## Requisitos
- Python 3.x, dependencias en `requirements.txt`.
- Autenticación de Google Cloud (ADC): `gcloud auth application-default login`.
- APIs habilitadas: Vertex AI, BigQuery, Cloud Vision.

## Modelos
- Embeddings: `text-multilingual-embedding-002` (multilingüe, optimizado para español).
- Generación: `gemini-2.5-flash` (alternativa: `gemini-2.5-pro`).
- SDK: `google-genai` (el SDK moderno y soportado para Vertex AI).
