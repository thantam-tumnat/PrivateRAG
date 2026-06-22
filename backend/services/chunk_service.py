def chunk_text(text, size=500, overlap=100):

    # overlap ต้องน้อยกว่า size เสมอ
    # ไม่งั้น step <= 0 แล้ว loop จะวนไม่รู้จบ
    if overlap >= size:
        raise ValueError(
            "overlap ต้องน้อยกว่า size"
        )

    # ก้าวทีละ (size - overlap) เพื่อให้แต่ละ chunk
    # เหลื่อมกับ chunk ก่อนหน้าเป็นระยะ overlap
    step = size - overlap

    chunks = []

    for i in range(0, len(text), step):
        chunks.append(
            text[i:i+size]
        )

    return chunks