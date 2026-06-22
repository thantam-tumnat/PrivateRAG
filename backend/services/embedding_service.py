"""
Embedding Service

หน้าที่:
- แปลงข้อความเป็น Vector
- ใช้สำหรับ Semantic Search
"""
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "intfloat/multilingual-e5-small"
)

# model ตระกูล E5 ต้องเติม prefix หน้าข้อความ
# - "query: "   สำหรับคำถาม
# - "passage: " สำหรับเนื้อหา/chunk
# ไม่ใส่ = ค้นหาแม่นน้อยลง
def embed_texts(texts, prefix="passage"):

    prefixed = [
        f"{prefix}: {t}"
        for t in texts
    ]

    return model.encode(prefixed)