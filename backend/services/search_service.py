"""
Search Service

- เปรียบเทียบ Query Vector กับ Chunk Vectors
- คำนวณคะแนนความใกล้เคียง (Similarity Score)

ใช้สำหรับทดลอง Semantic Search
ก่อนนำไปใช้กับ FAISS
"""
from sentence_transformers import util

def search(query_vector, chunk_vectors):

    scores = util.cos_sim(
        query_vector,
        chunk_vectors
    )

    return scores