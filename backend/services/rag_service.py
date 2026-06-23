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
from services.faiss_service import create_index, search_index, add_to_index
from services.llm_service import ask_llm


# เก็บ index + chunks ไว้ในแรม
# build ครั้งเดียวตอน startup แล้วใช้ซ้ำทุก request
_index = None
_chunks = [] 

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

    # กรณีไฟล์ไม่มีไฟล์ 
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ PDF ที่ {pdf_path}"
        )

    text = extract_text(pdf_path)

    _chunks = chunk_text(
        text,
        size=500,
        overlap=100
    )

    # กรณีเอกสารว่าง
    if not _chunks:
        raise ValueError(
            "อ่านข้อความจาก PDF ไม่ได้ (เอกสารว่างหรือเป็นไฟล์รูปภาพ)"
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

    # กันคำถามว่าง: ถ้าไม่ส่งคำถามมา ไม่ต้องไปค้นหา/ถาม LLM ให้เปลือง
    if not query or not query.strip():
        return {
            "context": [],
            "answer": "กรุณาพิมพ์คำถาม"
        }

    # ใช้ index ในแรม (ไม่ส่ง chunks)
    context = retrieve_context(
        query=query,
        top_k=2
    )

    # กัน context ว่าง: ถ้าค้นไม่เจอ chunk เลย อย่าส่งเข้า LLM
    # เพราะมันจะเดาคำตอบมั่ว ๆ (hallucinate) ทั้งที่ไม่มีข้อมูล
    if not context:
        return {
            "context": [],
            "answer": "ไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร"
        }

    answer = ask_llm(
        query,
        "\n\n".join(context)
    )

    return {
        "context": context,
        "answer": answer
    }


# อ่าน PDF ใหม่ 1 ไฟล์ -> chunk -> embed -> เติมเข้า index ในแรม
def add_document(pdf_path):
    global _index, _chunks

    text = extract_text(pdf_path)

    new_chunks = chunk_text(
        text,
        size=500,
        overlap=100
    )

    if not new_chunks:
        raise ValueError(
            "อ่านข้อความจาก PDF ไม่ได้ (เอกสารว่างหรือเป็นไฟล์รูปภาพ)"
        )

    vectors = embed_texts(
        new_chunks,
        prefix="passage"
    )

    if _index is None:
        _index = create_index(vectors)   # ยังไม่มี index -> สร้างใหม่
    else:
        add_to_index(_index, vectors)    # มีแล้ว -> เติมเข้าของเดิม

    _chunks.extend(new_chunks)

    return len(new_chunks)