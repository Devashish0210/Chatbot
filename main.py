from fastapi import FastAPI, HTTPException, Depends, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pytesseract
from PIL import Image
import io
from transformers import pipeline
import utils
import os


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

users_db = {}
contexts: List[str] = []


class RegisterInput(BaseModel):
    username: str
    email: str
    password: str
    name: str


class LoginInput(BaseModel):
    username: str
    password: str


@app.post("/register")
def register_user(user: RegisterInput):
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already exists")
    hashed_password = utils.hash_password(user.password)
    users_db[user.username] = {
        "email": user.email,
        "name": user.name,
        "password": hashed_password,
    }
    return {"message": "User registered successfully"}


@app.post("/login")
def login_user(credentials: LoginInput):
    user = users_db.get(credentials.username)
    if not user or not utils.verify_password(credentials.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = utils.create_access_token({"sub": credentials.username})
    return {"access_token": token, "token_type": "bearer"}


@app.post("/chat")
async def chat_answer(text: str = Form(None), file: UploadFile = File(None)):
    global contexts

    if file:
        try:
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Uploaded file is not an image.")
            
            file_content = await file.read()
            image = Image.open(io.BytesIO(file_content))

            extracted_text = pytesseract.image_to_string(image)
            if not extracted_text.strip():
                raise HTTPException(status_code=400, detail="No text extracted from the uploaded image.")

            contexts.append(extracted_text.strip())
            return {"response": f"Image processed and text extracted: {extracted_text.strip()}"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

    if text and contexts:
        try:
            full_context = " ".join(contexts)
            logger.info(f"Answering question based on context. Question: {text}")
            qa_result = qa_pipeline(question=text, context=full_context)
            return {"response": qa_result["answer"]}
        except Exception as e:
            logger.error(f"Error during QA processing: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error during QA processing: {str(e)}")
    
    return {"response": "I don't have any context to answer from. Please upload an image first."}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=port)
