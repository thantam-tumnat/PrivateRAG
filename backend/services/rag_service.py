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


# เก็บ index + chunks ไว้ในแรม
# build ครั้งเดียวตอน startup แล้วใช้ซ้ำทุก request
_index = None
_chunks = None

# สร้าง FAISS index เก็บไว้ในแรม
def build_index():
    """
    โหลด PDF -> chunk -> embed -> สร้าง FAISS index
    เรียกครั้งเดียวตอน server เริ่ม แล้วเก็บผลไว้ในแรม
    """
    global _index, _chunks

    pdf_path = (
        Path(__file__).parent.parent
        / "data"
        / "thantam_tumnat_resume.pdf"
    )

    text = extract_text(pdf_path)

    _chunks = chunk_text(
        text,
        size=500,
        overlap=100
    )

    chunk_vectors = embed_texts(
        _chunks,
        prefix="passage"
    )

    _index = create_index(chunk_vectors)

    # ส่งจำนวน chunks กลับไปแสดงตอน startup ว่า index พร้อมแล้ว
    return len(_chunks)

# หาข้อมูลที่เกี่ยวข้องกับ query จาก chunks 
def retrieve_context(
    query,
    chunks=None,
    top_k=2
):
    """
    หา chunk ที่ใกล้ query มากที่สุด top_k ชิ้น

    - ถ้าไม่ส่ง chunks มา -> ใช้ index ที่ build ไว้ในแรม (production)
    - ถ้าส่ง chunks มา    -> สร้าง index ใหม่จาก chunks นั้น (demo)
    """

    if chunks is None:
        # production: ใช้ index ที่ build ไว้แล้ว
        if _index is None:
            raise RuntimeError(
                "index ยังไม่ถูกสร้าง เรียก build_index() ก่อน"
            )

        index = _index #_index ถูกสร้างตอน startup ด้วย build_index() แล้วเก็บไว้ในแรม
        chunks = _chunks #_chunks ถูกสร้างตอน startup ด้วย build_index() แล้วเก็บไว้ในแรม

    else:
        # demo: build index จาก chunks ที่ส่งมาทุกครั้ง
        chunk_vectors = embed_texts(
            chunks,
            prefix="passage"
        )

        index = create_index(chunk_vectors)

    query_vector = embed_texts(
        [query],
        prefix="query"
    )
    #indices คือ list ของตำแหน่ง chunk ที่ใกล้เคียงกับ query มากที่สุด 
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

    # ใช้ index ในแรม (ไม่ส่ง chunks)
    context = retrieve_context(
        query=query,
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
