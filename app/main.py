from datetime import datetime
from typing import List
import pickle
import zipfile
import shutil
import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import fitz
import pytesseract


app = FastAPI()


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


# Create endpoint to classify documents
@app.post("/classify")
def classify_zip(request: BodyRequest):
    input_zip: str = request.input_zip
    output_folder: str = request.output_folder

    temp_extract_path = "temp_documents"
    os.makedirs(temp_extract_path, exist_ok=True)

    # Extract files
    with zipfile.ZipFile(input_zip, "r") as zip_ref:
        zip_ref.extractall(temp_extract_path)

    # Process anyd classiffy
    documents = process_documents(temp_extract_path)
    classifications = classify_documents(documents)
    organize_documents(classifications, output_folder)

    # Clean up the temp files
    shutil.rmtree(temp_extract_path)

    return JSONResponse(
        status_code=200,
        content={
            "message": "Classifications completed successfully",
            "results": classifications
        }
    )
