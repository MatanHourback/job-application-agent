from dotenv import load_dotenv
load_dotenv()
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
    Here is the candidate's resume:
    {resume_context}

    Here is the job description:
    {jd_context}

    List the skills, experiences, and qualifications the job requires that are 
    MISSING or WEAK in the resume. Interpret broadly — consider transferable skills,
    implied experience, and certification requirements.
    Be specific and use the exact terminology from the job description.
    """)
    chain = skill_gap_prompt | llm
    #Invoke the chain using the contexts that were just retrieved and sending said prompt to the llm
    return chain.invoke({"resume_context": resume_context, "jd_context": jd_context}).content

def generate_cover_letter(job_description):
    resume_context = get_context("skills and experience", RESUME_COLLECTION)
    jd_context = get_context("required skills", JD_COLLECTION)
    cover_letter_prompt = ChatPromptTemplate.from_template("""
    You are an expert career coach writing a cover letter.

    Here is the candidate's resume:
    {resume_context}

    Here is the job description:
    {jd_context}

    Write a concise 3 paragraph cover letter tailored to this specific role.
    Do NOT include placeholder text like [Your Name] or [Company Name].
    Do NOT include a header or address block.
    Start directly with "Dear Hiring Manager," and focus on relevant experience.
    Keep it under 250 words.
    CRITICAL: Only mention experience, skills, and accomplishments that are 
    explicitly stated in the resume. Do NOT fabricate or exaggerate credentials.
    If the candidate lacks key qualifications, focus on transferable skills instead.
    """)
    chain = cover_letter_prompt | llm
    #Invoke the chain using the contexts that were just retrieved and sending said prompt to the llm
    return chain.invoke({"resume_context": resume_context, "jd_context": jd_context}).content