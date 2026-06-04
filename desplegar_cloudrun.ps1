# =================================================================
#  desplegar_cloudrun.ps1
#  Despliega el Asistente RAG (app.py) en Google Cloud Run.
#
#  Cloud Run ESCALA A CERO: sin costo fijo, solo pagas por uso.
#
#  REQUISITO PREVIO (una sola vez): autenticar el gcloud CLI
#  (distinto de las credenciales ADC de las librerias):
#
#      & "C:\Users\Usuario\google-cloud-sdk\bin\gcloud.cmd" auth login
#
#  Luego ejecuta este script:
#      powershell -ExecutionPolicy Bypass -File .\desplegar_cloudrun.ps1
# =================================================================

$ErrorActionPreference = "Stop"
$GCLOUD  = "C:\Users\Usuario\google-cloud-sdk\bin\gcloud.cmd"
$PROJECT = "project-a0134db0-3990-4ec2-bc3"
$REGION  = "us-central1"
$SERVICE = "asistente-compras-publicas"

Write-Host "==> Proyecto: $PROJECT | Region: $REGION | Servicio: $SERVICE"

# 1) Habilitar las APIs necesarias para construir y desplegar.
Write-Host "==> Habilitando APIs (run, cloudbuild, artifactregistry)..."
& $GCLOUD services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com --project $PROJECT

# 2) Permisos: la cuenta de servicio en ejecucion de Cloud Run necesita:
#    - Vertex AI (embeddings + Gemini)            -> roles/aiplatform.user
#    - BigQuery (ejecutar consultas + leer datos) -> roles/bigquery.jobUser, roles/bigquery.dataViewer
#    - Cloud Vision API (OCR de PDFs escaneados). Vision no tiene un rol
#      dedicado para invocar la anotacion; el acceso se concede habilitando
#      la API + el permiso de consumo de servicios (serviceusage.services.use),
#      incluido en roles/serviceusage.serviceUsageConsumer.
$PROJECT_NUMBER = (& $GCLOUD projects describe $PROJECT --format="value(projectNumber)")
$RUNTIME_SA = "$PROJECT_NUMBER-compute@developer.gserviceaccount.com"
Write-Host "==> Otorgando roles a la cuenta de ejecucion: $RUNTIME_SA"
$ROLES = @(
    "roles/aiplatform.user",
    "roles/bigquery.jobUser",
    "roles/bigquery.dataViewer",
    "roles/serviceusage.serviceUsageConsumer"  # necesario para consumir Cloud Vision (OCR)
)
foreach ($rol in $ROLES) {
    & $GCLOUD projects add-iam-policy-binding $PROJECT `
        --member="serviceAccount:$RUNTIME_SA" --role="$rol" --condition=None | Out-Null
    Write-Host "    + $rol"
}

# 2.1) Asegurar que la API de Vision este habilitada antes de desplegar.
Write-Host "==> Habilitando Cloud Vision API (OCR)..."
& $GCLOUD services enable vision.googleapis.com --project $PROJECT

# 3) Desplegar desde el codigo fuente (Cloud Build construye la imagen del Dockerfile).
#    --allow-unauthenticated = acceso publico. Para restringir a usuarios internos,
#    quita esa bandera y administra accesos con IAM (roles/run.invoker).
Write-Host "==> Desplegando a Cloud Run (esto puede tardar unos minutos)..."
& $GCLOUD run deploy $SERVICE `
    --source . `
    --project $PROJECT `
    --region $REGION `
    --platform managed `
    --allow-unauthenticated `
    --memory 1Gi `
    --cpu 1 `
    --timeout 120 `
    --set-env-vars "GOOGLE_CLOUD_PROJECT=$PROJECT"

Write-Host ""
Write-Host "==> LISTO. URL del servicio:"
& $GCLOUD run services describe $SERVICE --project $PROJECT --region $REGION --format="value(status.url)"
