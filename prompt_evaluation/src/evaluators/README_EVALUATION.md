# Prompt Evaluation

This tool evaluates different prompts and generation types against questions from a CSV file, and logs the results to Weights & Biases (wandb).

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-wandb.txt
```

2. Sign up for a Weights & Biases account at [wandb.ai](https://wandb.ai) if you don't have one already.

3. Log in to Weights & Biases:
```bash
wandb login
```

## Usage

You can run the evaluation script by directly modifying the parameters in the main.py file:

```python
# In src/evaluators/main.py
if __name__ == "__main__":
    main(
        csv_file='qa_pairs_filtered.csv',
        num_questions=10,
        prompts=['base', 'medium', 'advanced'],
        generation_types=['llama'],
        wandb_project='prompt-evaluation',
        output_dir='assets/csv/evaluation_results'
    )
```

Then run the script:

```bash
python -m src.evaluators.main
```

Alternatively, you can import and use the functions directly in your own code:

```python
from src.evaluators.main import main

main(
    csv_file='qa_pairs_filtered.csv',
    num_questions=5,
    prompts=['base'],
    generation_types=['llama', 'openai'],
    wandb_project='prompt-evaluation',
    output_dir='assets/csv/evaluation_results'
)
```

### Configuration Parameters

- `csv_file`: Path to the CSV file containing questions and answers (default: qa_pairs_filtered.csv)
- `num_questions`: Number of questions to evaluate (default: 10)
- `prompts`: List of prompts to evaluate (default: ['base', 'medium', 'advanced'])
- `generation_types`: List of generation types to evaluate (default: ['llama'])
- `wandb_project`: Weights & Biases project name (default: 'prompt-evaluation')
- `wandb_entity`: Weights & Biases entity name (default: None)
- `output_dir`: Directory to save evaluation results (default: 'assets/csv/evaluation_results')

## How It Works

The evaluation process follows these steps:

1. **Model Initialization**: For each generation type (e.g., 'llama', 'openai'), the script initializes the appropriate models using the `initialize_models` function from `generator_modules.py`. This ensures consistent model initialization across the evaluation process.

2. **Question Processing**: For each question in the CSV file, the script:
   - Retrieves relevant context using the retriever
   - Builds a prompt using the specified prompt template
   - Generates an answer using the initialized model
   - Evaluates the answer against the ground truth

3. **Answer Generation**: The script directly uses the initialized models to generate answers:
   - For Llama models, it calls the model with the prompt and parameters
   - For OpenAI models, it uses the chat completions API

4. **Evaluation Metrics**: The script calculates various metrics to evaluate the quality of the generated answers (see below).

5. **Logging**: Results are logged to Weights & Biases and saved to a CSV file.

## CSV File Format

The CSV file should have the following columns:
- `question`: The question to be answered
- `id`: A unique identifier for the question
- `answer`: The ground truth answer
- `context`: The context for the question (optional)

## Evaluation Metrics

The script evaluates the generated answers using the following metrics:

1. **Cosine Similarity**: Measures the semantic similarity between the generated answer and the ground truth.
2. **ROUGE Scores**: Measures the overlap of n-grams between the generated answer and the ground truth.
   - ROUGE-1: Unigram overlap
   - ROUGE-2: Bigram overlap
   - ROUGE-L: Longest common subsequence
3. **Answer Quality**: Uses OpenAI's API to evaluate the answer quality based on:
   - Clarity: Is the generated answer clear, well-structured, and easy to understand?
   - Relevancy: How well does the generated answer address the core concepts present in the ground truth?
   - Factual Correctness: Are the facts in the generated answer accurate when compared to the ground truth?

## Results

The evaluation results are saved to a CSV file in the specified output directory and logged to Weights & Biases. You can view the results in the Weights & Biases dashboard.

### Viewing Results

1. Go to [wandb.ai](https://wandb.ai) and log in
2. Navigate to your project (default: prompt-evaluation)
3. View the results in the dashboard

## Customization

You can customize the evaluation by:

1. Adding new prompts in `src/prompts/prompt_templates.py`
2. Adding new generation types in `src/evaluators/generator_modules.py` and `src/evaluators/wandb_evaluation.py`
3. Adding new evaluation metrics in `src/evaluators/evaluation_funcs.py` 