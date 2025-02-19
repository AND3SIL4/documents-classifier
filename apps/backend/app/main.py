from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
import uuid
from datetime import datetime
from typing import List
import pickle
import zipfile
import shutil
import os

from pydantic import BaseModel
import fitz
import pytesseract


app = FastAPI()


# Allow cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"]
)


# Create a base model to make the body request
class BodyRequest(BaseModel):
    input_zip: str
    output_folder: str


# Classification model trained on google colab
MODEL_PATH = 'app/machine_learning/model.pkl'
VECTORIZER_PATH = 'app/machine_learning/vectorizer.pkl'


# Main end point that show the status and information
@app.get("/")
def root():
    time = datetime.now().strftime("%m-%d-%Y %H:%M:%S")
    return JSONResponse(
        status_code=200,
        content={
            "status": "working",
            "time": time,
            "version": "0.0.1",
            "author": "Felipe Silva"
        }
    )


# Load model and verctorizer trained
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)


# Extract text from PDF file
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text")
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")

    return text


# Extract text from image
def extract_text_from_image(image_path: str) -> str:
    try:
        text = pytesseract.image_to_string(image_path)
    except Exception as e:
        print(f"Error extrating text from {image_path}: {e}")

    return text


# Process documents and extract text
def process_documents(input_folder: str) -> List[dict]:
    documents = []

    for root, _, files in os.walk(input_folder):
        for file in files:
            file_path = os.path.join(root, file)
            text = ""
            if file.lower().endswith(".pdf"):
                text = extract_text_from_pdf(file_path)
            elif file.lower().endswith((".png", ".jpg", ".jpeg")):
                text = extract_text_from_image(file_path)

            if text:
                documents.append({"file": file_path, "text": text})

    return documents


# Classify documents
def classify_documents(documents: List[dict]):
    results = {}

    for doc in documents:
        text_vectorized = vectorizer.transform([doc["text"]])
        prediction = model.predict(text_vectorized)[0]
        results[doc["file"]] = prediction

    return results


# Move files to folders predicted
def organize_documents(classiications: dict, output_folder: str):
    for file_path, category in classiications.items():
        category_folder = os.path.join(output_folder, category)
        os.makedirs(category_folder, exist_ok=True)
        shutil.move(file_path, os.path.join(
            category_folder, os.path.basename(file_path)))


# Carpeta base para almacenar archivos temporales
BASE_OUTPUT_FOLDER = "classified_temp"
os.makedirs(BASE_OUTPUT_FOLDER, exist_ok=True)


@app.post("/classify")
async def classify_zip(file: UploadFile = File(...)):
    # Crear una carpeta única para esta solicitud
    request_id = str(uuid.uuid4())
    temp_folder = os.path.join(BASE_OUTPUT_FOLDER, request_id)
    os.makedirs(temp_folder, exist_ok=True)

    # Guardar el archivo ZIP temporalmente
    temp_zip_path = os.path.join(temp_folder, file.filename)
    with open(temp_zip_path, "wb") as f:
        f.write(await file.read())

    try:
        # Extraer y clasificar documentos
        extract_folder = os.path.join(temp_folder, "extracted")
        os.makedirs(extract_folder, exist_ok=True)
        with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_folder)

        # Procesar y clasificar documentos
        documents = process_documents(extract_folder)
        classifications = classify_documents(documents)

        # Mover los archivos clasificados a una carpeta de salida
        output_folder = os.path.join(temp_folder, "classified_output")
        os.makedirs(output_folder, exist_ok=True)
        organize_documents(classifications, output_folder)

        # Crear un archivo ZIP solo con la carpeta de salida clasificada
        output_zip_path = os.path.join(temp_folder, "classified_files.zip")
        with zipfile.ZipFile(output_zip_path, "w") as zipf:
            for root, _, files in os.walk(output_folder):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    arcname = os.path.relpath(file_path, output_folder)
                    zipf.write(file_path, arcname)

        return JSONResponse(
            status_code=200,
            content={
                "message": "Classifications completed successfully",
                "request_id": request_id,
                "download_link": f"/download/{request_id}"
            }
        )
    except Exception as e:
        # Limpiar en caso de error
        shutil.rmtree(temp_folder, ignore_errors=True)
        raise HTTPException(
            status_code=500, detail=f"Error processing file: {str(e)}"
        )


@app.get("/download/{request_id}")
async def download_classified_files(
    request_id: str, background_tasks: BackgroundTasks
):
    # Verificar que la carpeta de la solicitud exista
    temp_folder = os.path.join(BASE_OUTPUT_FOLDER, request_id)
    output_zip_path = os.path.join(temp_folder, "classified_files.zip")

    if not os.path.exists(output_zip_path):
        raise HTTPException(
            status_code=404, detail="Files not found or already downloaded"
        )

    # Agregar la tarea de limpieza en segundo plano
    background_tasks.add_task(cleanup_temp_folder, temp_folder)

    # Devolver el archivo ZIP
    return FileResponse(
        output_zip_path,
        media_type="application/zip",
        filename="classified_files.zip"
    )


# Función para eliminar la carpeta temporal
def cleanup_temp_folder(temp_folder: str):
    try:
        shutil.rmtree(temp_folder, ignore_errors=True)
        print(f"Carpeta temporal eliminada: {temp_folder}")
    except Exception as e:
        print(f"Error eliminando la carpeta temporal: {e}")
