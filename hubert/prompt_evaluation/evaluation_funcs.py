from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
import numpy as np
from openai import OpenAI
from langfuse import Langfuse
from src.config.settings import settings


model = SentenceTransformer('all-MiniLM-L6-v2')

langfuse = Langfuse(
    secret_key=settings.LANGFUSE_SECRET_KEY,
    public_key=settings.LANGFUSE_PUBLIC_KEY,
    host=settings.LANGFUSE_HOST)


def calculate_cosine_similarity(generated: str, ground_truth: str, model: SentenceTransformer) -> float:
    """Calculate semantic similarity between generated answer and ground truth answer."""
    gen_emb = model.encode([generated])[0]
    truth_emb = model.encode([ground_truth])[0]
    # Normalize vectors and compute cosine similarity
    sim = (gen_emb @ truth_emb) / (np.linalg.norm(gen_emb) * np.linalg.norm(truth_emb))
    return sim


def calculate_rouge_scores(generated: str, ground_truth: str) -> dict:
    """Calculate ROUGE scores between generated and ground truth answers.
    
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metrics measure text similarity:
    
    - ROUGE-1: Measures the overlap of unigrams (single words) between the texts.
               Good for measuring content overlap at the word level.
               
    - ROUGE-2: Measures the overlap of bigrams (two consecutive words) between the texts.
               Better captures local word order and phrasing than ROUGE-1.
               
    - ROUGE-L: Measures the Longest Common Subsequence (LCS) between the texts.
               Captures longer-range sentence structure while allowing for some gaps.
               
    Each ROUGE metric returns three values:
    - precision: How many of the n-grams in the generated text appear in the reference text
    - recall: How many of the n-grams in the reference text appear in the generated text
    - fmeasure: The harmonic mean of precision and recall (F1 score)
    
    Returns:
        dict: Dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(generated, ground_truth)
    return scores


def calculate_answer_quality(question: str, answer: str, ground_truth: str) -> float:
    """Calculate answer quality score based on criteria."""
    evaluation_prompt = """You are an objective evaluator. Assess the generated answer against the ground truth on a scale of 0-10 for each criterion below. Provide a JSON response with scores and brief explanations.

    Criteria:
    1. Clarity (0-10): Is the generated answer clear, well-structured, and easy to understand?
    2. Relevancy (0-10): How well does the generated answer address the core concepts present in the ground truth?
    3. Factual Correctness (0-10): Are the facts in the generated answer accurate when compared to the ground truth?

    Format your response as a JSON object:
    {
        "clarity": {"score": X},
        "relevancy": {"score": X},
        "factual_correctness": {"score": X},
        "overall_score": {"score": X, "explanation": "..."}
    } """

    openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

    response = openai_client.chat.completions.create(
        model="o1-mini",  
        messages=[
            {"role": "user", "content": evaluation_prompt},
            {"role": "user", "content": f"Question:\n{question}\n\nAnswer:\n{answer}\n\nGround Truth:\n{ground_truth}"}
        ]
    )
    
    return response.choices[0].message.content

