from fastapi import FastAPI

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