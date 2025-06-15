# The main function for prompt evaluation
# Should take in a question, retrieve the context using the retriever, 
# Combine with the prompt that we defined earlier, and then generate an answer using the LLM
# Should save the results to a CSV file

from hubert.prompt_evaluation.evaluators.retriever import Retriever
from hubert.config import settings
from hubert.prompt_evaluation.evaluators.generator_modules import initialize_models
from hubert.prompt_evaluation.evaluators.wandb_evaluation import run_evaluation

EMBEDDING_MODEL = settings.embedding_model

def main(
    csv_file: str = 'qa_pairs_filtered.csv',
    prompts: list = ['base', 'medium', 'advanced'],
    generation_types: list = ['llama'],
    wandb_project: str = 'prompt-evaluation',
    wandb_entity: str = None,
    output_dir: str = 'assets/csv/evaluation_results'
):
    """
    Main function to run the evaluation.
    
    This function evaluates different prompts and generation types against questions from a CSV file,
    and logs the results to Weights & Biases (wandb). It uses the generator_modules.initialize_models
    function to initialize the models for each generation type.
    
    The evaluation process:
    1. Initializes models for each generation type
    2. For each question, retrieves relevant context
    3. Generates an answer using the specified prompt and model
    4. Evaluates the answer against the ground truth
    5. Logs results to wandb and saves to CSV
    
    Each prompt-generation type combination will be logged as a separate wandb run,
    making it easier to compare results across different configurations.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing questions and answers
    num_questions : int
        Number of questions to evaluate
    prompts : list
        List of prompts to evaluate
    generation_types : list
        List of generation types to evaluate (e.g., 'llama', 'openai')
    wandb_project : str
        Weights & Biases project name
    wandb_entity : str
        Weights & Biases entity name
    output_dir : str
        Directory to save evaluation results
    """
    print(f"Starting evaluation with the following configuration:")
    print(f"- CSV file: {csv_file}")
    print(f"- Prompts: {prompts}")
    print(f"- Generation types: {generation_types}")
    print(f"- WandB project: {wandb_project}")
    print(f"- Output directory: {output_dir}")
    print(f"- Each prompt-generation type combination will be logged as a separate wandb run")
    
    # Test model initialization to ensure it works
    print("Testing model initialization...")
    for generation_type in generation_types:
        print(f"Initializing models for {generation_type}")
        llm, embedding_generator, reranker_model = initialize_models(model_type=generation_type)
        print(f"Successfully initialized models for {generation_type}")
        
        # Clean up resources
        if hasattr(embedding_generator, 'cleanup'):
            embedding_generator.cleanup()
    
    # Run the evaluation
    output_file = run_evaluation(
        csv_file=csv_file,
        prompts=prompts,
        generation_types=generation_types,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        output_dir=output_dir
    )
    
    print(f"Evaluation completed. Results saved to {output_file}")
    print(f"Individual results for each prompt-generation type combination have been saved to separate CSV files.")
    print(f"Check your WandB dashboard at https://wandb.ai/dashboard to view the results for each combination.")

if __name__ == "__main__":
    # You can modify these values directly here instead of using command-line arguments
    main(
        csv_file='qa_pairs_filtered.csv',
        prompts=['base', 'medium', 'advanced'],
        generation_types=['llama'],
        wandb_project='prompt-evaluation',
        output_dir='assets/csv/evaluation_results/new'
    )
