from fastapi import FastAPI
from services.rag_service import ask_question

app = FastAPI()


@app.get("/")
def root():

    return {
        "message": "Private RAG API"
    }


@app.get("/ask")
def ask(query: str):

    return ask_question(query)