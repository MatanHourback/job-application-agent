import anthropic

client = anthropic.Anthropic()

def score_faithfulness(resume: str, cover_letter: str) -> float:
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=10,  #only need a number back
        system="You are a strict, precise evaluator. You respond only with a single decimal number between 0 and 1. You never add explanation or commentary.", #locks claude into an evaluator role before even reading the prompt and less likely to deviate from simply returning a number
        messages=[
            {"role": "user", "content": f"""
            You are evaluating an AI-generated cover letter for faithfulness.

            Read the resume and cover letter below. Score how well every claim 
            in the cover letter is grounded in the resume.

            1.0 = every claim is directly supported by the resume
            0.5 = some claims are unsupported or exaggerated  
            0.0 = most claims are fabricated or not in the resume

            Resume: {resume}
            Cover Letter: {cover_letter}

            Respond with ONLY a single decimal number between 0 and 1. No other text.
            """}
        ]
    )
    try:
        return float(response.content[0].text.strip())
    except ValueError:
        print(f"Warning: unexpected judge response: {response.content[0].text}")
        return 0.0

def score_relevance(job_description: str, cover_letter: str) -> float:
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=10,  #only need a number back
        system="You are a strict, precise evaluator. You respond only with a single decimal number between 0 and 1. You never add explanation or commentary.", #locks claude into an evaluator role before even reading the prompt and less likely to deviate from simply returning a number
        messages=[
            {"role": "user", "content": f"""
            You are evaluating an AI-generated cover letter for relevance.

            Read the job description and cover letter below. Score how relevant every claim in the cover letter is to the job description.

            1.0 = every claim directly addresses the job description
            0.5 = some claims are irrelevant or superfluous  
            0.0 = most claims are irrelevant or not in the job description

            Job description: {job_description}
            Cover Letter: {cover_letter}

            Respond with ONLY a single decimal number between 0 and 1. No other text.
            """}
        ]
    )
    try:
        return float(response.content[0].text.strip())
    except ValueError:
        print(f"Warning: unexpected judge response: {response.content[0].text}")
        return 0.0