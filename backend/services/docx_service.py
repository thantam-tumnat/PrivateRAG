"""
DOCX Service

- อ่านไฟล์ Word (.docx)
- ดึงข้อความออกจากเอกสาร
"""

from docx import Document


def extract_text(docx_path):

    doc = Document(docx_path)

    # ต่อทุกย่อหน้าด้วยขึ้นบรรทัดใหม่
    text = "\n".join(
        p.text
        for p in doc.paragraphs
    )

    return text
