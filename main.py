from fastapi import FastAPI, HTTPException, Depends, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pytesseract
from PIL import Image
import io
from transformers import pipeline
import utils

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
async def chat_answer(text: str = Form(...), file: UploadFile = File(None)):
    """
    Endpoint for handling user input and returning responses.
    - Extracts text from uploaded images using Tesseract OCR.
    - Uses Hugging Face pipeline for answering questions.
    """
    global contexts

    if file:
        try:
            image = Image.open(io.BytesIO(await file.read()))
            extracted_text = pytesseract.image_to_string(image)
            contexts.append(extracted_text.strip())
            return {"response": f"Image uploaded and text extracted: {extracted_text.strip()}"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

    if text and contexts:
        full_context = " ".join(contexts)
        qa_result = qa_pipeline(question=text, context=full_context)
        return {"response": qa_result["answer"]}

    return {"response": "I don't have any context to answer from. Please upload an image first."}
