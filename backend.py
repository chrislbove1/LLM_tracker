from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tiktoken

# Optional parsers for different file types
from io import BytesIO
from typing import Optional
import docx  # pip install python-docx
import PyPDF2  # pip install PyPDF2

app = FastAPI()

# Allow your frontend (e.g., http://localhost:5500) to call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------

class TextRequest(BaseModel):
    text: str
    model: Optional[str] = "gpt-4o"  # any tokenizer supported by tiktoken


# ---------- Helper: get tokenizer ----------

def get_tokenizer(model_name: str = "gpt-4o"):
    """
    Returns a tiktoken tokenizer compatible with a given model.
    Fall back to cl100k_base if the model isn't known.
    """
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return enc


# ---------- Endpoint 1: raw text ----------

@app.post("/tokenize-text")
async def tokenize_text(payload: TextRequest):
    enc = get_tokenizer(payload.model)
    tokens = enc.encode(payload.text)
    return {
        "model": payload.model,
        "token_count": len(tokens),
        "tokens": tokens,  # you can drop this if you only need the count
    }


# ---------- Helper: extract text from different file types ----------

def extract_text_from_file(filename: str, content: bytes) -> str:
    """
    Extracts visible text from common file types.
    Extend this as needed (.pptx, .xlsx, etc.).
    """
    name = filename.lower()

    # Plain text-like files
    if name.endswith((".txt", ".md", ".csv", ".json", ".py", ".js", ".html", ".css")):
        try:
            return content.decode("utf-8", errors="ignore")
        except Exception:
            return content.decode("latin-1", errors="ignore")

    # DOCX
    if name.endswith(".docx"):
        file_like = BytesIO(content)
        doc = docx.Document(file_like)
        paragraphs = [p.text for p in doc.paragraphs]
        return "\n".join(paragraphs)

    # PDF
    if name.endswith(".pdf"):
        text = []
        file_like = BytesIO(content)
        reader = PyPDF2.PdfReader(file_like)
        for page in reader.pages:
            text.append(page.extract_text() or "")
        return "\n".join(text)

    # Fallback: try to interpret as text
    try:
        return content.decode("utf-8", errors="ignore")
    except Exception:
        return ""


# ---------- Endpoint 2: file upload ----------

@app.post("/tokenize-file")
async def tokenize_file(
    file: UploadFile = File(...),
    model: str = Form("gpt-4o"),
):
    content = await file.read()
    raw_text = extract_text_from_file(file.filename, content)

    enc = get_tokenizer(model)
    tokens = enc.encode(raw_text)

    return {
        "filename": file.filename,
        "model": model,
        "byte_size": len(content),
        "text_length": len(raw_text),
        "token_count": len(tokens),
    }
