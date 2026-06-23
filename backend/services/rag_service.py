"""
RAG Service

หน้าที่:
- โหลด PDF
- สร้าง Chunks
- Retrieve Context
- ส่งเข้า LLM
"""

from pathlib import Path
from services.chunk_service import chunk_text
from services.embedding_service import embed_texts
from services.faiss_service import create_index, search_index, add_to_index
from services.llm_service import ask_llm
from services.pdf_service import extract_text as extract_pdf
from services.docx_service import extract_text as extract_docx

# เก็บ index + chunks ไว้ในแรม
# build ครั้งเดียวตอน startup แล้วใช้ซ้ำทุก request
_index = None
_chunks = [] 

# นามสกุลไฟล์ที่รองรับ
SUPPORTED = {".pdf", ".docx"}


# อ่านไฟล์ 1 ไฟล์ (เลือกตัวอ่านตามนามสกุล) -> chunk -> embed -> เติมเข้า index
# ใช้ร่วมกันทั้ง build_index (วนทั้ง folder) และ add_document (ไฟล์เดียว)
def _add_file(file_path):
    global _index, _chunks

    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    # เลือกตัวอ่านตามนามสกุล
    if suffix == ".pdf":
        text = extract_pdf(file_path)
    elif suffix == ".docx":
        text = extract_docx(file_path)
    else:
        raise ValueError(
            f"ยังไม่รองรับไฟล์นามสกุล {suffix}"
        )

    new_chunks = chunk_text(
        text,
        size=500,
        overlap=100
    )

    # กรณีอ่านข้อความไม่ได้ (เอกสารว่าง/ไฟล์สแกน)
    if not new_chunks:
        raise ValueError(
            "อ่านข้อความจากไฟล์ไม่ได้ (เอกสารว่างหรือเป็นไฟล์รูปภาพ)"
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


# สแกนทุกไฟล์ที่รองรับใน data/ -> เติมเข้า index
# เรียกครั้งเดียวตอน server เริ่ม
def build_index():

    data_dir = Path(__file__).parent.parent / "data"

    # หาทุกไฟล์ที่นามสกุลรองรับในโฟลเดอร์ data
    files = [
        f for f in data_dir.iterdir()
        if f.suffix.lower() in SUPPORTED
    ]

    if not files:
        raise FileNotFoundError(
            f"ไม่พบไฟล์ที่รองรับใน {data_dir}"
        )

    for f in files:
        try:
            _add_file(f)
        except ValueError:
            # ไฟล์ไหนอ่านไม่ได้ = ข้าม
            continue

    # ถ้าทุกไฟล์อ่านไม่ได้เลย -> index ว่าง เตือนไว้
    if not _chunks:
        raise ValueError(
            "อ่านไฟล์ใน data/ ไม่ได้สักไฟล์"
        )

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

    # กรณีส่ง blank text มา
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

# อัปโหลดไฟล์ใหม่ 1 ไฟล์ -> เติมเข้า index (ใช้ logic เดียวกับตอน startup)
def add_document(file_path):
    return _add_file(file_path)