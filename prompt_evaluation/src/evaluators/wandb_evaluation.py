import os
import time
import pandas as pd
import wandb
import json
from datetime import datetime
from tqdm import tqdm
import logging
from typing import List, Dict, Any, Optional

from src.evaluators.retriever import Retriever
from src.config import settings
from src.evaluators.config import settings as evaluator_settings
from src.prompts.prompt_templates import PromptFactory
from src.evaluators.evaluation_funcs import (
    calculate_cosine_similarity, 
    calculate_rouge_scores, 
    calculate_answer_quality,
    model as embedding_model
)
from src.evaluators.generator_modules import initialize_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_wandb(project: str, entity: Optional[str] = None, config: Dict[str, Any] = None, run_name: str = None):
    """Initialize Weights & Biases."""
    name = run_name if run_name else f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return wandb.init(
        project=project,
        entity=entity,
        config=config,
        job_type="evaluation",
        name=name
    )

def load_questions(csv_file: str, num_questions: Optional[int] = None) -> pd.DataFrame:
    """Load questions from CSV file."""
    try:
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df)} questions from {csv_file}")
        return df.head(num_questions) if num_questions is not None else df
    except Exception as e:
        logger.error(f"Error loading questions from {csv_file}: {e}")
        raise

def evaluate_answer(question: str, generated_answer: str, ground_truth: str) -> Dict[str, Any]:
    """Evaluate the generated answer against the ground truth."""
    results = {}
    
    # Calculate cosine similarity
    try:
        results["cosine_similarity"] = calculate_cosine_similarity(generated_answer, ground_truth, embedding_model)
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        results["cosine_similarity"] = None
    
    # Calculate ROUGE scores
    try:
        rouge_scores = calculate_rouge_scores(generated_answer, ground_truth)
        results.update({
            'rouge1_precision': rouge_scores['rouge1'].precision,
            'rouge1_recall': rouge_scores['rouge1'].recall,
            'rouge1_fmeasure': rouge_scores['rouge1'].fmeasure,
            'rouge2_precision': rouge_scores['rouge2'].precision,
            'rouge2_recall': rouge_scores['rouge2'].recall,
            'rouge2_fmeasure': rouge_scores['rouge2'].fmeasure,
            'rougeL_precision': rouge_scores['rougeL'].precision,
            'rougeL_recall': rouge_scores['rougeL'].recall,
            'rougeL_fmeasure': rouge_scores['rougeL'].fmeasure
        })
    except Exception as e:
        logger.error(f"Error calculating ROUGE scores: {e}")
        results.update({
            'rouge1_precision': None, 'rouge1_recall': None, 'rouge1_fmeasure': None,
            'rouge2_precision': None, 'rouge2_recall': None, 'rouge2_fmeasure': None,
            'rougeL_precision': None, 'rougeL_recall': None, 'rougeL_fmeasure': None
        })
    
    # Calculate answer quality
    try:
        quality_eval_raw = calculate_answer_quality(question, generated_answer, ground_truth)
        
        # Try to parse the JSON response
        if isinstance(quality_eval_raw, str):
            # Remove any markdown formatting if present
            quality_eval_raw = quality_eval_raw.replace('```json\n', '').replace('\n```', '')
            quality_eval = json.loads(quality_eval_raw)
        else:
            quality_eval = quality_eval_raw
        
        results.update({
            'clarity_score': quality_eval.get('clarity', {}).get('score'),
            'relevancy_score': quality_eval.get('relevancy', {}).get('score'),
            'factual_correctness_score': quality_eval.get('factual_correctness', {}).get('score'),
            'overall_score': quality_eval.get('overall_score', {}).get('score') if isinstance(quality_eval.get('overall_score'), dict) else quality_eval.get('overall_score'),
            'overall_explanation': quality_eval.get('overall_score', {}).get('explanation') if isinstance(quality_eval.get('overall_score'), dict) else None
        })
    except Exception as e:
        logger.error(f"Error calculating quality evaluation: {e}")
        results.update({
            'clarity_score': None,
            'relevancy_score': None,
            'factual_correctness_score': None,
            'overall_score': None,
            'overall_explanation': None,
            'quality_eval_error': str(e)
        })
    
    return results

def generate_answer(
    question: str,
    context: str,
    llm,
    prompt_builder,
    generation_type: str,
    max_tokens: int = 256,
    temperature: float = 0.1
) -> Dict[str, Any]:
    """Generate an answer using the LLM."""
    # Build the prompt
    prompt_text = prompt_builder.build_prompt(user_question=question, context=context)
    
    # Generate response based on model type
    if generation_type == "llama":
        response = llm(
            prompt=prompt_text,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            echo=False
        )
        
        # Extract the generated text from the response
        if isinstance(response, dict) and "choices" in response:
            generated_text = response["choices"][0]["text"]
        else:
            generated_text = response.get("choices", [{}])[0].get("text", str(response))
            
        return {
            "choices": [{"text": generated_text}],
            "context": context,
            "prompt": prompt_text
        }
    
    elif generation_type == "openai":
        response = llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        generated_text = response.choices[0].message.content
        
        return {
            "choices": [{"text": generated_text}],
            "context": context,
            "prompt": prompt_text
        }
    
    else:
        raise ValueError(f"Unsupported generation type: {generation_type}")

def run_evaluation(
    csv_file: str = 'qa_pairs_filtered.csv',
    num_questions: Optional[int] = None,
    prompts: List[str] = ['base', 'medium', 'advanced'],
    generation_types: List[str] = ['llama', 'openai'],
    wandb_project: str = 'prompt-evaluation',
    wandb_entity: Optional[str] = None,
    output_dir: str = 'assets/csv/evaluation_results'
):
    """Run the evaluation with the given configuration."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load questions
    questions_df = load_questions(csv_file, num_questions)
    
    # Initialize retriever
    retriever = Retriever(
        top_k=10, 
        use_reranker=True, 
        use_hybrid_search=True, 
        hybrid_alpha=0.6,
        embedding_provider="openai",
        embedding_model=evaluator_settings.embedding_model
    )
    
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Iterate over prompts and generation types
    for prompt_type in prompts:
        for generation_type in generation_types:
            logger.info(f"Evaluating prompt: {prompt_type}, generation type: {generation_type}")
            
            # Initialize a separate wandb run for each prompt-generation type combination
            run_name = f"eval_{prompt_type}_{generation_type}"
            run = initialize_wandb(
                project=wandb_project,
                entity=wandb_entity,
                config={
                    "prompt_type": prompt_type,
                    "generation_type": generation_type,
                    "num_questions": num_questions,
                    "csv_file": csv_file
                },
                run_name=run_name
            )
            
            # Initialize models using generator_modules
            logger.info(f"Initializing models for {generation_type}")
            llm, embedding_generator, reranker_model = initialize_models(model_type=generation_type)
            
            # Initialize prompt builder
            prompt_builder = PromptFactory.create_prompt(prompt_type)
            
            # Results for this specific prompt-generation combination
            combination_results = []
            
            # Iterate over questions
            for _, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc=f"{prompt_type}_{generation_type}"):
                question = row['question']
                ground_truth = row['answer']
                question_id = row['id']
                
                # Generate answer
                start_time = time.time()
                try:
                    # Retrieve context
                    doc_ids, scores, query_time, contents_list = retriever.retrieve(question, "page_embeddings_alpha")
                    context = "\n\n---\n\n".join(contents_list) if contents_list else ""
                    
                    # Generate answer
                    rag_response = generate_answer(
                        question=question,
                        context=context,
                        llm=llm,
                        prompt_builder=prompt_builder,
                        generation_type=generation_type,
                        max_tokens=256,
                        temperature=0.1
                    )
                    
                    generated_answer = rag_response['choices'][0]['text']
                    context = rag_response['context']
                    prompt = rag_response.get('prompt', '')
                    generation_time = time.time() - start_time
                    
                    # Evaluate answer
                    evaluation_results = evaluate_answer(question, generated_answer, ground_truth)
                    
                    # Create result entry
                    result_entry = {
                        "id": question_id,
                        "question": question,
                        "ground_truth": ground_truth,
                        "retrieved_context": context,
                        "generated_answer": generated_answer,
                        "prompt": prompt,
                        "prompt_type": prompt_type,
                        "generation_type": generation_type,
                        "generation_time": generation_time
                    }
                    result_entry.update(evaluation_results)
                    
                    # Log to wandb
                    log_data = {
                        "question_id": question_id,
                        "question": question,
                        "generation_time": generation_time,
                        "cosine_similarity": evaluation_results.get("cosine_similarity"),
                        "rouge1_fmeasure": evaluation_results.get("rouge1_fmeasure"),
                        "rouge2_fmeasure": evaluation_results.get("rouge2_fmeasure"),
                        "rougeL_fmeasure": evaluation_results.get("rougeL_fmeasure"),
                        "clarity_score": evaluation_results.get("clarity_score"),
                        "relevancy_score": evaluation_results.get("relevancy_score"),
                        "factual_correctness_score": evaluation_results.get("factual_correctness_score"),
                        "overall_score": evaluation_results.get("overall_score")
                    }
                    
                    # Create a table for detailed text comparison
                    comparison_table = wandb.Table(columns=["Question", "Ground Truth", "Generated Answer"])
                    comparison_table.add_data(question, ground_truth, generated_answer)
                    
                    # Log both metrics and the table
                    wandb.log(log_data)
                    wandb.log({f"text_comparison_{question_id}": comparison_table})
                    
                    combination_results.append(result_entry)
                    all_results.append(result_entry)
                    
                except Exception as e:
                    logger.error(f"Error generating answer for question {question_id}: {e}")
                    continue
            
            # Save results for this combination to CSV
            combination_df = pd.DataFrame(combination_results)
            combination_output_file = os.path.join(output_dir, f"eval_{prompt_type}_{generation_type}.csv")
            combination_df.to_csv(combination_output_file, index=False)
            logger.info(f"Results for {prompt_type}_{generation_type} saved to {combination_output_file}")
            
            # Upload combination results file to wandb
            wandb.save(combination_output_file)
            
            # Create a summary table with results for this combination
            summary_table = wandb.Table(dataframe=combination_df)
            wandb.log({"evaluation_summary": summary_table})
            
            # Finish this wandb run
            wandb.finish()
            
            # Clean up resources
            if hasattr(embedding_generator, 'cleanup'):
                embedding_generator.cleanup()
    
    # Save all results to a combined CSV
    all_results_df = pd.DataFrame(all_results)
    all_output_file = os.path.join(output_dir, f"eval_all_results.csv")
    all_results_df.to_csv(all_output_file, index=False)
    logger.info(f"All evaluation results saved to {all_output_file}")
    
    return all_output_file

if __name__ == "__main__":
    # Default configuration when run directly
    run_evaluation() 