import fitz  # PyMuPDF
import re
from pathlib import Path
 
 
def parse_pdf_resume(file_bytes: bytes) -> str:
    """
    Extract clean text from a resume PDF.
    Args:
        file_bytes: Raw bytes of the uploaded PDF file
    Returns:
        Cleaned text string from the resume
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text_parts = []
 
    for page in doc:
        text = page.get_text("text")  # "text" mode preserves reading order
        text_parts.append(text)
 
    doc.close()
    raw_text = "\n".join(text_parts)
    return clean_text(raw_text)
 
 
def parse_job_description(jd_text: str) -> str:
    """
    Clean and normalize a pasted job description.
    Args:
        jd_text: Raw job description string from user input
    Returns:
        Cleaned job description string
    """
    return clean_text(jd_text)
 
 
def clean_text(text: str) -> str:
    """
    Remove excessive whitespace and normalize line breaks.
    """
    # Collapse multiple blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    return text.strip()
 
 
# --- Quick test (run this file directly to verify) ---
if __name__ == "__main__":
    sample_jd = """
    Software Engineer - AI/ML
 
    We are looking for a passionate engineer to join our AI team.
 
    Requirements:
    - 3+ years Python experience
    - Experience with LLMs and prompt engineering
    - Familiarity with vector databases (Pinecone, Weaviate, ChromaDB)
    - Strong communication skills
    """
    print("=== Job Description ===")
    print(parse_job_description(sample_jd))