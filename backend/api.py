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