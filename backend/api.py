from fastapi import FastAPI
import fitz
app = FastAPI()

@app.get("/")
def root():
    return {"message":"Hello"}

@app.get("/hello")
def hello(name: str):
    return {
        "message": f"Hello {name}"
    }

@app.get("/hi")
def hi(name: str):
    return {
        "message": f"Hi {name}",        
        "alert": "This is an alert message"
    }

@app.get("/readpdf")
def read_pdf():
    doc = fitz.open("thantam_tumnat_resume.pdf")

    text = ""

    for page in doc:
        text += page.get_text()

    return {
        "content": text
    }

@app.get("/chunks")
def chunks():

    doc = fitz.open("thantam_tumnat_resume.pdf")

    text = ""

    for page in doc:
        text += page.get_text()

    chunks = chunk_text(text)

    return {
        "chunk_count": len(chunks),
        "chunks": chunks
    }

def chunk_text(text, size=500):

    chunks = []

    for i in range(0, len(text), size):
        chunks.append(text[i:i+size])

    return chunks