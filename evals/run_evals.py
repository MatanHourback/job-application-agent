import json
from app.agent import analyze_skill_gaps, generate_cover_letter
from app.embeddings import index_document, clear_collection, RESUME_COLLECTION, JD_COLLECTION
from evals.test_cases import test_cases
from evals.judge import score_faithfulness, score_relevance
from dotenv import load_dotenv

load_dotenv()

def score_recall(known_gaps: list, agent_output: str) -> float:
    # check how many known gaps appear in the agent output
    found = 0
    for gap in known_gaps:
        if gap.lower() in agent_output.lower():
            found += 1
    recall = found / len(known_gaps)
    return recall
    

def run_evals():
    results = []

    for test in test_cases:
        # Clear stale data before each test
        clear_collection(RESUME_COLLECTION)
        clear_collection(JD_COLLECTION)
    
        # Now index fresh data
        index_document(test["resume"], RESUME_COLLECTION)
        index_document(test["job_description"], JD_COLLECTION)

        print(f"\nRunning {test['id']} ({test['sector']})...")
        index_document(test["resume"], RESUME_COLLECTION)
        index_document(test["job_description"], JD_COLLECTION)
        skill_gaps = analyze_skill_gaps(test["job_description"])
        cover_letter = generate_cover_letter(test["job_description"])

        recall = score_recall(test["known_skill_gaps"], skill_gaps)
        faithfulness = score_faithfulness(test["resume"], cover_letter)
        relevance = score_relevance(test["job_description"], cover_letter)

        results.append({
            "id": test["id"],
            "sector": test["sector"],
            "recall": recall,
            "faithfulness": faithfulness,
            "relevance": relevance,
        })

        # Print summary table for this test case
        print(f"  Recall:      {recall:.2f}")
        print(f"  Faithfulness:{faithfulness:.2f}")
        print(f"  Relevance:   {relevance:.2f}")

    #Print average test scores
    print(f"Average recall score:      {sum(r['recall'] for r in results) / len(results):.2f}")
    print(f"Average faithfulness score:{sum(r['faithfulness'] for r in results) / len(results):.2f}")
    print(f"Average relevance score:   {sum(r['relevance'] for r in results) / len(results):.2f}")

    # Save results to evals/results.json
    with open("evals/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to evals/results.json")
    
    

if __name__ == "__main__":
    run_evals()