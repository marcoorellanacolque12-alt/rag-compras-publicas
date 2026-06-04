# Imagen del Asistente RAG de Contrataciones Publicas (FastAPI + Gemini + BigQuery)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Codigo de la app (la API + el motor RAG + extraccion para el modulo de analisis)
COPY app.py responder.py extraccion_texto.py ./

# Cloud Run inyecta el puerto en $PORT (default 8080).
ENV PORT=8080
EXPOSE 8080

# Arranque. Shell form para expandir $PORT.
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}
