from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.parser import parse_pdf_resume, parse_job_description
from app.embeddings import index_document, RESUME_COLLECTION, JD_COLLECTION
from app.agent import analyze_skill_gaps, generate_cover_letter
import os

 
app = FastAPI(title="Job Application Agent")
 
# Serve the frontend HTML
app.mount("/static", StaticFiles(directory="frontend"), name="static")
 
 
# --- Request / Response models ---
 
class AnalyzeRequest(BaseModel):
    job_description: str
 
class AnalyzeResponse(BaseModel):
    skill_gaps: str
    cover_letter: str
 
 
# --- In-memory flag: tracks whether a resume has been indexed this session ---
resume_indexed = {"status": False}
 
 
# --- Routes ---
 
@app.get("/")
def root():
    return FileResponse("frontend/index.html")
 
 
@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """
    Accepts a PDF resume, parses it, and indexes it into ChromaDB.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
 
    file_bytes = await file.read()
 
    # Step 1: Parse PDF → clean text
    resume_text = parse_pdf_resume(file_bytes)
 
    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF. Is it a scanned image?")
 
    # Step 2: Chunk + embed + store in ChromaDB
    index_document(resume_text, RESUME_COLLECTION)
    resume_indexed["status"] = True
 
    return {
        "message": "Resume uploaded and indexed successfully.",
        "character_count": len(resume_text),
    }
 
 
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Takes a job description, indexes it, then runs the RAG chains
    to produce skill gap analysis and a tailored cover letter.
    """
    if not resume_indexed["status"]:
        raise HTTPException(status_code=400, detail="Please upload a resume first.")
 
    if not request.job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")
 
    # Step 1: Parse + index the job description
    jd_text = parse_job_description(request.job_description)
    index_document(jd_text, JD_COLLECTION)
 
    return AnalyzeResponse(
        skill_gaps=analyze_skill_gaps(jd_text),
        cover_letter=generate_cover_letter(jd_text)
    )
 
 
# --- Health check ---
@app.get("/health")
def health():
    return {"status": "ok"}