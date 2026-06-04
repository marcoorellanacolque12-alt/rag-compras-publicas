"""
crear_datalake.py
-------------------------------------------------------------------
Crea la infraestructura inicial (Data Lake) en Google Cloud Storage
para un sistema RAG de contrataciones publicas.

- Crea el bucket 'repositorio-compras-publicas'.
  Si el nombre ya esta tomado a nivel GLOBAL, agrega un sufijo
  numerico hasta encontrar uno disponible.
- Crea la jerarquia de 5 carpetas (prefijos) normativas.

Requisitos:
    pip install google-cloud-storage
    Autenticacion previa (ADC):
        gcloud auth application-default login

Uso:
    python crear_datalake.py
-------------------------------------------------------------------
"""

from google.cloud import storage
from google.api_core import exceptions

# ===================== CONFIGURACION =====================
PROJECT_ID = "project-a0134db0-3990-4ec2-bc3"
BUCKET_BASE_NAME = "repositorio-compras-publicas"
LOCATION = "US"  # Ubicacion del bucket. Cambia a "southamerica-east1" (Sao Paulo)
                 # o "us-central1" segun tu region de preferencia.
STORAGE_CLASS = "STANDARD"

# Estructura documental normativa obligatoria (prefijos / "carpetas")
CARPETAS_NORMATIVAS = [
    "leyes_y_reglamentos",
    "directivas",
    "opiniones",
    "resoluciones_tribunal",
    "documentos_orientacion",
]
# ========================================================


def obtener_bucket_unico(client, base_name):
    """
    Devuelve un objeto Bucket ya creado con un nombre globalmente unico.
    Intenta con el nombre base; si esta tomado, prueba base-1, base-2, ...
    """
    intento = 0
    while True:
        nombre = base_name if intento == 0 else f"{base_name}-{intento}"
        try:
            print(f"-> Intentando crear el bucket: '{nombre}' ...")
            bucket = client.bucket(nombre)
            bucket.storage_class = STORAGE_CLASS
            nuevo = client.create_bucket(bucket, location=LOCATION)
            print(f"   [OK] Bucket creado: gs://{nuevo.name} (location={nuevo.location})")
            return nuevo
        except exceptions.Conflict:
            # 409: el nombre ya existe (puede ser de otra cuenta o de la tuya).
            print(f"   [!] '{nombre}' ya esta tomado a nivel global. Probando otro sufijo...")
            intento += 1
            if intento > 50:
                raise RuntimeError("No se encontro un nombre de bucket disponible tras 50 intentos.")
        except exceptions.Forbidden as e:
            print("   [X] Permiso denegado. Revisa que tu cuenta tenga el rol "
                  "'Storage Admin' en el proyecto y que la API de Cloud Storage este habilitada.")
            raise e


def crear_estructura_carpetas(bucket, carpetas):
    """
    En GCS no existen carpetas reales; se simulan con objetos vacios
    terminados en '/'. Esto crea los 5 prefijos normativos.
    """
    print("\n-> Creando la estructura documental normativa (prefijos):")
    for carpeta in carpetas:
        # Se usa un marcador estandar para que la 'carpeta' sea visible en la consola.
        marcador = f"{carpeta}/"
        blob = bucket.blob(marcador)
        if not blob.exists():
            blob.upload_from_string("", content_type="application/x-directory")
        print(f"   [OK] /{carpeta}")


def main():
    print("=" * 60)
    print(" CREACION DE DATA LAKE - CONTRATACIONES PUBLICAS (RAG)")
    print("=" * 60)
    print(f"Proyecto GCP: {PROJECT_ID}\n")

    client = storage.Client(project=PROJECT_ID)

    bucket = obtener_bucket_unico(client, BUCKET_BASE_NAME)
    crear_estructura_carpetas(bucket, CARPETAS_NORMATIVAS)

    print("\n" + "=" * 60)
    print(" INFRAESTRUCTURA LISTA")
    print("=" * 60)
    print(f"Bucket final : gs://{bucket.name}")
    print("Rutas normativas:")
    for carpeta in CARPETAS_NORMATIVAS:
        print(f"   gs://{bucket.name}/{carpeta}/")
    print("\nGuarda el nombre del bucket; lo necesitaras en 'subida_masiva.py'.")


if __name__ == "__main__":
    main()
