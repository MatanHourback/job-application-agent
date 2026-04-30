from app.embeddings import load_vector_store, RESUME_COLLECTION, JD_COLLECTION
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Using manual retrieval over a retriever chain in order to merge two vector stores into a single prompt

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
#temperature=0 means deterministic (important for evals): same input = same output


def get_context(query, collection_name, k=4):
    #load the vector store for this collection
    vector_store = load_vector_store(collection_name)
    #search it with the query
    results = vector_store.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in results])

def analyze_skill_gaps(job_description):
    resume_context = get_context("skills and experience", RESUME_COLLECTION)
    jd_context = get_context("required skills", JD_COLLECTION)
    skill_gap_prompt = ChatPromptTemplate.from_template("""
    You are an expert career coach.
    Here is the candidate's resume:
    {resume_context}

    Here is the job description:
    {jd_context}

    List the skills and experiences the job requires that are 
    MISSING or WEAK in the resume. Be specific and concise.
    """)
    chain = skill_gap_prompt | llm
    #Invoke the chain using the contexts that were just retrieved and sending said prompt to the llm
    return chain.invoke({"resume_context": resume_context, "jd_context": jd_context}).content

def generate_cover_letter(job_description):
    resume_context = get_context("skills and experience", RESUME_COLLECTION)
    jd_context = get_context("required skills", JD_COLLECTION)
    cover_letter_prompt = ChatPromptTemplate.from_template("""
    You are an expert career coach.
    Here is the candidate's resume:
    {resume_context}

    Here is the job description:
    {jd_context}

    Write a cover letter for this resume that would fit well with the job description.
    """)
    chain = cover_letter_prompt | llm
    #Invoke the chain using the contexts that were just retrieved and sending said prompt to the llm
    return chain.invoke({"resume_context": resume_context, "jd_context": jd_context}).content