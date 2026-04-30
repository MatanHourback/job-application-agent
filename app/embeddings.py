from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os
 
 
# --- Configuration ---
CHUNK_SIZE = 500        # Characters per chunk (good for resumes - sections aren't huge)
CHUNK_OVERLAP = 50      # Overlap so context isn't lost at chunk boundaries
RESUME_COLLECTION = "resume"
JD_COLLECTION = "job_description"
CHROMA_DIR = "./chroma_db"  # Local folder where ChromaDB saves data
 
 
def get_embeddings():
    """
    Returns the embedding model.
    Reads OPENAI_API_KEY from environment automatically.
    """
    return OpenAIEmbeddings(model="text-embedding-3-small")
 
 
def chunk_text(text: str) -> list[str]:
    """
    Split a document into overlapping chunks for embedding.
    Uses RecursiveCharacterTextSplitter which tries to split on
    paragraphs → sentences → words (in that order) to keep context intact.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],  # Try these splits in order
    )
    return splitter.split_text(text)
 
 
def index_document(text: str, collection_name: str) -> Chroma:
    """
    Chunk a document and store embeddings in ChromaDB.
    Args:
        text: Full document text
        collection_name: "resume" or "job_description"
    Returns:
        Chroma vector store instance (used later for retrieval)
    """
    chunks = chunk_text(text)
    print(f"[Embeddings] '{collection_name}' split into {len(chunks)} chunks")
 
    embeddings = get_embeddings()
 
    # Chroma will save to disk at CHROMA_DIR
    # If a collection already exists with this name, it will be overwritten
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_DIR,
        metadatas=[{"source": collection_name, "chunk_index": i} for i in range(len(chunks))],
    )
 
    print(f"[Embeddings] Indexed {len(chunks)} chunks into '{collection_name}'")
    return vector_store
 
 
def load_vector_store(collection_name: str) -> Chroma:
    """
    Load an existing ChromaDB collection from disk.
    Call this after index_document() to retrieve later.
    """
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )
 
 
def retrieve(query: str, collection_name: str, k: int = 4) -> list[str]:
    """
    Retrieve the top-k most relevant chunks for a query.
    Args:
        query: The question or search string
        collection_name: Which collection to search ("resume" or "job_description")
        k: Number of chunks to return
    Returns:
        List of relevant text chunks
    """
    vector_store = load_vector_store(collection_name)
    results = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]
 
 
# --- Quick test ---
if __name__ == "__main__":
    sample_resume = """
    Jane Doe | jane@email.com | github.com/janedoe
 
    EXPERIENCE
    Software Engineer at Acme Corp (2021–2024)
    - Built REST APIs using FastAPI and Python
    - Worked with PostgreSQL and Redis for data storage
    - Deployed services on AWS using Docker and Kubernetes
 
    SKILLS
    Python, FastAPI, SQL, Docker, AWS, Git
 
    EDUCATION
    B.S. Computer Science, State University, 2021
    """
 
    # Index the resume
    index_document(sample_resume, RESUME_COLLECTION)
 
    # Test retrieval
    results = retrieve("What cloud platforms has this person used?", RESUME_COLLECTION)
    print("\n=== Retrieval Test ===")
    for i, chunk in enumerate(results):
        print(f"[Chunk {i+1}] {chunk[:200]}")
 