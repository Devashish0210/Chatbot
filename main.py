from fastapi import FastAPI, HTTPException, Depends, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pytesseract
from PIL import Image
import io
from transformers import pipeline
import utils
import logging
import os

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust origins if needed for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tesseract OCR Path (Ensure it's configured correctly for your environment)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Hugging Face QA pipeline initialization
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# In-memory user database
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
        logger.warning(f"Username already exists: {user.username}")
        raise HTTPException(status_code=400, detail="Username already exists")
    hashed_password = utils.hash_password(user.password)
    users_db[user.username] = {
        "email": user.email,
        "name": user.name,
        "password": hashed_password,
    }
    logger.info(f"User registered: {user.username}")
    return {"message": "User registered successfully"}


@app.post("/login")
def login_user(credentials: LoginInput):
    user = users_db.get(credentials.username)
    if not user or not utils.verify_password(credentials.password, user["password"]):
        logger.warning(f"Invalid login attempt for username: {credentials.username}")
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = utils.create_access_token({"sub": credentials.username})
    logger.info(f"Login successful for username: {credentials.username}")
    return {"access_token": token, "token_type": "bearer"}


@app.post("/chat")
async def chat_answer(text: str = Form(None), file: UploadFile = File(None)):
    global contexts

    # Process uploaded image
    if file:
        try:
            if not file.content_type.startswith("image/"):
                logger.warning(f"Invalid file type: {file.content_type}")
                raise HTTPException(status_code=400, detail="Uploaded file is not an image.")
            
            # Read and process image
            file_content = await file.read()
            logger.info(f"File content size: {len(file_content)} bytes")
            image = Image.open(io.BytesIO(file_content))
            logger.info(f"Image format: {image.format}, size: {image.size}")

            # Extract text using pytesseract
            extracted_text = pytesseract.image_to_string(image)
            if not extracted_text.strip():
                logger.warning("No text extracted from the uploaded image.")
                raise HTTPException(status_code=400, detail="No text extracted from the uploaded image.")

            contexts.append(extracted_text.strip())
            logger.info(f"Text extracted and added to context: {extracted_text.strip()}")
            return {"response": f"Image processed and text extracted: {extracted_text.strip()}"}
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

    # Handle text input
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


# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 if PORT not set
    logger.info(f"Starting FastAPI server on port {port}")
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=port)
