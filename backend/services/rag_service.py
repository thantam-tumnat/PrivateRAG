"""
RAG Service

หน้าที่:
- โหลด PDF
- สร้าง Chunks
- Retrieve Context
- ส่งเข้า LLM
"""

from pathlib import Path

from services.pdf_service import extract_text
from services.chunk_service import chunk_text
from services.embedding_service import embed_texts
from services.faiss_service import create_index, search_index
from services.llm_service import ask_llm


def retrieve_context(
    query,
    chunks,
    top_k=2
):

    chunk_vectors = embed_texts(
        chunks,
        prefix="passage"
    )

    index = create_index(
        chunk_vectors
    )

    query_vector = embed_texts(
        [query],
        prefix="query"
    )

    distances, indices = search_index(
        index,
        query_vector,
        k=top_k
    )

    results = []

    for idx in indices[0]:

        results.append(
            chunks[idx]
        )

    return results


def ask_question(query):
    # โหลด PDF
    pdf_path = (
        Path(__file__).parent.parent
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