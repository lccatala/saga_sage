from contextlib import asynccontextmanager
from datetime import datetime
import os
import shutil
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import uvicorn

from generate_db import create_database
from query_db import ask_question


class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    context: Optional[list[str]] = None
    sources: Optional[list[str]] = None

class UploadResponse(BaseModel):
    filename: str
    size: int
    upload_time: str


UPLOAD_DIR = "uploads"
DB_DIR =  "chroma"

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    yield
    for filename in os.listdir(UPLOAD_DIR):
        os.remove(os.path.join(UPLOAD_DIR, filename))
    os.rmdir(UPLOAD_DIR)

app = FastAPI(
    lifespan=lifespan, 
    title="RAG API", 
    description="API for querying documents using RAG", 
    version="0.0.1")

def is_valid_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() == ".epub"

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    if file.filename is None:
        return HTTPException(status_code=400, detail="File has no filename")
    if not is_valid_file(file.filename):
        return HTTPException(status_code=400, detail="Only .epub files are allowed")

    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_stats = os.stat(file_path)
        upload_response = UploadResponse(
            filename=file.filename, 
            size=file_stats.st_size, 
            upload_time=datetime.now().isoformat())
        return upload_response
    except Exception as e:
        return HTTPException(status_code=500, detail=f"An error occured while uploading the file: {str(e)}")

@app.get("/files")
async def list_files():
    files = []
    for filename in os.listdir(UPLOAD_DIR):
        if is_valid_file(filename):
            file_path = os.path.join(UPLOAD_DIR, filename)
            stats = os.stat(file_path)
            files.append({
                "filename": filename,
                "size": stats.st_size,
                "upload_time": datetime.fromtimestamp(stats.st_mtime).isoformat()
            })
    return {"files": files}

@app.delete("/files/{filename}")
async def delete_file(filename: str):
    if not is_valid_file(filename):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        os.remove(file_path)
        return {"message": f"File {filename} successfully deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while deleting the file: {str(e)}")

@app.post("/ask", response_model=QuestionResponse)
async def ask(request: QuestionRequest):
    try:
        if not os.path.exists(UPLOAD_DIR) or not os.listdir(UPLOAD_DIR):
            raise HTTPException(status_code=400, detail="No documents have been uploaded yet")

        n_books = len(os.listdir(UPLOAD_DIR)) if os.path.exists(UPLOAD_DIR) else 0
        n_dbs = len(os.listdir(DB_DIR)) if os.path.exists(DB_DIR) else 0
        if n_books != n_dbs or n_dbs == 0: 
            create_database(UPLOAD_DIR, DB_DIR)

        answer_dict = ask_question(request.question, DB_DIR)
        question_response = QuestionResponse(
            answer=str(answer_dict["answer"]), 
            sources=list(answer_dict["sources"]))

        return question_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    load_dotenv()
    uvicorn.run(app, host="0.0.0.0", port=8000)
