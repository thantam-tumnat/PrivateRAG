import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File

from services.rag_service import build_index, ask_question, add_document
DATA_DIR = Path(__file__).parent / "data"

@asynccontextmanager
async def lifespan(app: FastAPI):
    count = build_index()
    print(f"Index พร้อมแล้ว: {count} chunks")
    
    # build_index() เสร็จถึงจะ yield(เปิดapi)
    yield
  


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():

    return {
        "message": "Private RAG API"
    }


@app.get("/ask")
def ask(query: str):

    return ask_question(query)

@app.post("/upload")
def upload(file: UploadFile = File(...)):

    save_path = DATA_DIR / file.filename

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    count = add_document(save_path)

    return {
        "filename": file.filename,
        "chunks_added": count
    }