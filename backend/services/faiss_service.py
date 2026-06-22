"""
FAISS Service

หน้าที่:
- สร้าง Vector Index
- ค้นหา Top-K Chunks ที่เกี่ยวข้อง
"""
# Top-k คือการค้นหาสิ่งที่เกี่ยวข้องมากที่สุด k ชิ้น เช่น ถ้า k=3 ก็จะได้ผลลัพธ์ที่เกี่ยวข้องมากที่สุด 3 ชิ้น
import faiss
import numpy as np

    
# สร้าง FAISS Index จาก Chunk Vectors
def create_index(vectors):
    
    # แปลง List ของ Vectors เป็น numpy array ชื่อตัวแปรเดิม
    vectors = np.array(
        vectors,
        dtype="float32"
    )

    # จำนวนมิติของ Vector
    dimension = vectors.shape[1]  
    # เช่น แปลงจาก vectors เป็น np ได้ผลลัพธ์ (4,384)  
    # แล้วจะเอาค่า 384 มาใช้เป็นมิติของ Vector ก็เลย กำหนด dimensionn = vectors.shape[1]  

    # สร้าง Index แบบ L2 Distance จากค่า dimension
    index = faiss.IndexFlatL2(
        dimension
    ) #เสมือนการจองตำแหน่งในหน่วยความจำสำหรับ Vector ที่มีมิติ 384

    # เพิ่ม Vector ทั้งหมดเข้า Index
    index.add(vectors) # เหมือน record ที่เก็บ index กับ vector หน้าตาประมาณว่า {index: vector} เช่น {0: [0.1, 0.2, 0.3], 1: [0.4, 0.5, 0.6]}

    return index 

# ฟังก์ชันค้นหา Top-K Vector ที่ใกล้ Query มากที่สุด
def search_index(
    index,
    query_vector,
    k=3
):

    # แปลง Query เป็น numpy array faiss ต้องการข้อมูลในรูปแบบ numpy array ที่มีชนิดข้อมูลเป็น float32
    query_vector = np.array(
        query_vector,
        dtype="float32"
    )

    # ค้นหา Top-K คือ
    # เทียบ Query Vector กับ Vector ใน Index เพื่อหาค่า Distance ที่ใกล้เคียงที่สุด 
    # และคืนค่า Distance กับ Indices ของ Vector ทั้งหมดที่ใกล้เคียงที่สุด
    distances, indices = index.search(
        query_vector,
        k
    )
    
    return distances, indices