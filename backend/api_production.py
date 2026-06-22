from contextlib import asynccontextmanager

from fastapi import FastAPI

from services.rag_service import build_index, ask_question


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
