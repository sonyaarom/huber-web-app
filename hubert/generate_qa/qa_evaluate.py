from src.config import settings
import openai
import time
import random
import re
from tqdm import tqdm
from openai import RateLimitError, AuthenticationError
from logging import getLogger
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

logging = getLogger(__name__)

api_key = settings.openai_api_key

if not api_key:
    raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

# Set up OpenAI client
client = openai.Client(api_key=api_key)

def is_non_specific_question(question):
    # List of patterns for very general, non-specific questions
    patterns = [
        r"^what is this about\??$",
        r"^what is the main topic\??$",
        r"^what (is|are) the main point(s)?\??$",
        r"^can you summarize this\??$",
        r"^what (is|are) the key takeaway(s)?\??$",
        r"^what is the main (point|idea|topic) of this (article|paper)\??$",
        r"^what information does the text provide\??$",
        r"^what is the main point of this paper\??$",
        r"^[Your specific question here]\??$",  # Placeholder for additional patterns
        # Patterns for detecting questions that refer to unspecified models or papers
        r"^what (does|do) the (model|paper|study) (propose|suggest|conclude)\??$",
        r"^how does the (model|paper|study) improve (on|over) existing (models|methods)\??$"
    ]

    # Convert the input question to lower case for case insensitive matching
    question = question.lower()
    
    # Check if the question matches any pattern
    return any(re.match(pattern, question) for pattern in patterns)

@retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(20), retry=retry_if_exception_type((openai.RateLimitError, openai.APIError)))
def evaluate_question(question):
    # Check if it's a non-specific question
    if is_non_specific_question(question):
        return {
            "Specificity": 1,  # Very low specificity
            "Realism": 1,     # Likely realistic, as these are common questions
            "Clarity": 1      # Usually very clear, even if not specific
        }

    evaluation_prompt = f"""
    You are an AI assistant that evaluates questions based on specificity, realism, and clarity.s
    Evaluate the following question based on three criteria using a scale of 1 to 5:
    1. **Specificity**: Does the question clearly identify a specific paper, model, or study, or specific topic, avoiding general references like 'the model' or 'the paper'? The question should mention specific details that clearly delineate which content is being queried, rather than asking broadly.
    2. **Realism**: Is the question realistic and aligned with what students might genuinely ask in an academic setting? The question should reflect practical and common-sense inquiries likely to arise during study or review.
    3. **Clarity**: Is the question clearly formulated, avoiding ambiguous language or phrasing that could confuse students? The question should be easy to understand and free from vague terms or complex structures.
    Rate each criterion from 1 to 5, where:
    1 - Very Poor
    2 - Poor
    3 - Fair
    4 - Good
    5 - Excellent
    Question: "{question}"
    Please provide a rating for each criterion along with a brief explanation.
    Specificity:
    Realism:
    Clarity:
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant that evaluates questions based on specificity, realism, and clarity."},
                {"role": "user", "content": evaluation_prompt}
            ],
            max_tokens=150,
            temperature=0.5,
        )
        response_text = response.choices[0].message.content.strip()
    except openai.RateLimitError:
        logging.warning("Rate limit reached. Retrying...")
        raise
    except openai.AuthenticationError:
        logging.error("Authentication failed. Please check your API key.")
        raise
    except Exception as e:
        logging.error(f"Error in API call: {e}")
        return None

    scores = {
        "Specificity": None,
        "Realism": None,
        "Clarity": None
    }
    for line in response_text.splitlines():
        for criterion in scores.keys():
            if f"{criterion}:" in line:
                try:
                    scores[criterion] = int(line.split(":")[1].strip().split()[0])
                except (ValueError, IndexError):
                    logging.error(f"Error parsing score for {criterion}")
    
    return scores

def process_questions(df, batch_size=1):
    evaluated_questions_df = df.copy()
    evaluated_questions_df['Specificity'] = None
    evaluated_questions_df['Realism'] = None
    evaluated_questions_df['Clarity'] = None
    evaluated_questions_df['Average Score'] = None
    evaluated_questions_df['Is Non-Specific'] = None

    for i in tqdm(range(0, len(evaluated_questions_df), batch_size), desc="Processing questions"):
        batch = evaluated_questions_df.iloc[i:i+batch_size]
        for index, row in batch.iterrows():
            question = row['question']
            if question is None:
                continue
            is_non_specific = is_non_specific_question(question)
            evaluated_questions_df.at[index, 'Is Non-Specific'] = is_non_specific
            
            scores = evaluate_question(question)
            if scores:
                valid_scores = [score for score in scores.values() if score is not None]
                for criterion, score in scores.items():
                    evaluated_questions_df.at[index, criterion] = score
                if valid_scores:
                    average_score = sum(valid_scores) / len(valid_scores)
                    evaluated_questions_df.at[index, 'Average Score'] = average_score
            
            # Implement exponential backoff with jitter
            time.sleep(random.uniform(5, 15))

    return evaluated_questions_df


if __name__ == "__main__":
    # Load and process QA pairs
    df = pd.read_csv('qa_pairs.csv')
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Convert NaN/None to empty string to avoid float errors
            question = str(row['question']) if pd.notna(row['question']) else ""
            answer = str(row['answer']) if pd.notna(row['answer']) else ""
            context = str(row['extracted_context']) if pd.notna(row['extracted_context']) else ""
            
            # Skip empty questions or answers
            if not question.strip() or not answer.strip():
                continue
                
            results.append({
                'question': question,
                'answer': answer, 
                'context': context
            })
        except Exception as e:
            logging.error(f"Error processing row {idx}: {str(e)}")
            continue
            
    df = pd.DataFrame(results)
    try:
        evaluated_questions_df = process_questions(df)

    # Save the DataFrame with evaluations to a CSV file
        evaluated_questions_df.to_csv('evaluated_questions_with_scores.csv', index=False)

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")