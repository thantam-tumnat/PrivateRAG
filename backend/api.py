"""
FastAPI Entry Point
- รับ HTTP Request
- เรียกใช้ Service ต่าง ๆ
- ส่ง Response กลับเป็น JSON
"""
from services.llm_service import ask_llm
from pathlib import Path
from fastapi import FastAPI
from services.pdf_service import extract_text
from services.chunk_service import chunk_text
from services.embedding_service import embed_texts
from services.search_service import search
from services.faiss_service import (
    create_index,
    search_index
)

from services.embedding_service import (
    embed_texts
)
from services.rag_service import (
    retrieve_context
)

app = FastAPI()


@app.get("/")
def root():

    return {
        "message": "Hello"
    }


@app.get("/hello")
def hello(name: str):

    return {
        "message": f"Hello {name}"
    }


@app.get("/readpdf")
def read_pdf():

    pdf_path = (
        Path(__file__).parent
        / "data"
        / "thantam_tumnat_resume.pdf"
    )

    text = extract_text(pdf_path)

    return {
        "content": text
    }


@app.get("/chunks")
def get_chunks():

    pdf_path = (
        Path(__file__).parent
        / "data"
        / "thantam_tumnat_resume.pdf"
    )

    text = extract_text(pdf_path)

    chunks = chunk_text(text)

    return {
        "chunk_count": len(chunks),
        "chunks": chunks
    }

@app.get("/embedding")
def embedding_demo():

    chunks = [
        "Python FastAPI Docker",
        "Java Spring Boot",
        "C# .NET WPF",
        "Python Django PostgreSQL"
    ]

    vectors = embed_texts(chunks)

    return {
        "chunk_count": len(chunks),
        "vector_dimension": len(vectors[0])
    }

@app.get("/search")
def search_demo():

    chunks = [
        "Python FastAPI Docker",
        "Java Spring Boot",
        "C# .NET WPF",
        "Python Django PostgreSQL"
    ]

    chunk_vectors = embed_texts(chunks)

    query_vector = embed_texts(
        ["Python"],
        prefix="query"
    )

    scores = search(
        query_vector,
        chunk_vectors
    )

    return {
        "scores": scores.tolist()
    }

@app.get("/faiss")
def faiss_demo():

    # จำลอง Chunks
    chunks = [
        "Python FastAPI Docker",
        "Java Spring Boot",
        "C# .NET WPF",
        "Python Django PostgreSQL"
    ]

    # แปลง Chunks เป็น Vectors
    chunk_vectors = embed_texts(
        chunks
    )

    # สร้าง FAISS Index
    index = create_index(
        chunk_vectors
    )

    # แปลงคำถามเป็น Vector
    query_vector = embed_texts(
        ["Python"],
        prefix="query"
    )

    # หา Top 2 Chunks
    distances, indices = search_index(
        index,
        query_vector,
        k=2
    )

    results = []

    # ดึงข้อความจริงกลับมา
    for idx in indices[0]:

        results.append(
            chunks[idx]
        )

    return {
        "results": results,
        "indices": indices.tolist(),
        "distances": distances.tolist()
    }

@app.get("/rag")
def rag_demo(query: str):

    pdf_path = (
        Path(__file__).parent
        / "data"
        / "thantam_tumnat_resume.pdf"
    )

    text = extract_text(pdf_path)

    chunks = chunk_text(text, size=500, overlap=100)
    
    context = retrieve_context(
        query=query,
        chunks=chunks,
        top_k=2
    )

    return {
        "query": query,
        "context": context
    }

@app.get("/ask")
def ask(query: str):

    pdf_path = (
        Path(__file__).parent
        / "data"
        / "thantam_tumnat_resume.pdf"
    )

    text = extract_text(
        pdf_path
    )

    chunks = chunk_text(
        text,
        size=500,
        overlap=100
    )

    context = retrieve_context(
        query=query,
        chunks=chunks,
        top_k=2
    )

    answer = ask_llm(
        query,
        "\n\n".join(context)
    )

    return {
        "context": context,
        "answer": answer
    }