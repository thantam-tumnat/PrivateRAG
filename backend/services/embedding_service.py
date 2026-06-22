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

def embed_texts(texts):

    return model.encode(texts)