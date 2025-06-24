# Evaluation Dashboard

A comprehensive UI for conducting retriever and prompt evaluations in the HuBer system.

## Features

### 🔍 Retriever Evaluation
- Test different embedding tables and retrieval strategies
- Configure hybrid search (vector + BM25) with adjustable alpha
- Enable/disable rerankers for improved precision
- Adjustable top-k results and other parameters
- Results logged to Weights & Biases

### 💬 Prompt Evaluation  
- Test different prompt templates (base, medium, advanced)
- Compare multiple language models (OpenAI, Llama)
- Evaluate answer quality with multiple metrics:
  - ROUGE scores (1, 2, L)
  - Cosine similarity
  - Clarity, relevancy, and factual correctness scores
- Results saved as CSV files and logged to Weights & Biases

### 🚀 Combined Evaluation
- Run both retriever and prompt evaluation in sequence
- Quick presets for common evaluation scenarios
- Comprehensive testing across all parameters

### 📋 Data Source Management
- Upload custom CSV files with validation
- Use default QA pairs from the assets folder
- CSV preview functionality
- Required columns: id, question, answer
- Optional columns: context

## Access Control

**Admin Only**: All evaluation functionality requires admin user privileges. Non-admin users will be redirected with an access denied message.

## CSV File Format

Your CSV files should have the following structure:

```csv
id,question,answer,context
1,"What is machine learning?","Machine learning is...", "ML context"
2,"How does AI work?","AI works by...", "AI context"
```

### Required Columns:
- `id`: Unique identifier for each QA pair
- `question`: The question to be answered
- `answer`: The ground truth answer for evaluation

### Optional Columns:
- `context`: Additional context information

## Usage

### 1. Access the Dashboard
Navigate to `/evaluation` (admin access required)

### 2. Configure Data Source
- Choose a default CSV file from the available options
- Or upload your own CSV file with the required format

### 3. Run Evaluations

#### Retriever Evaluation:
1. Select embedding tables to test
2. Configure parameters (top-k, hybrid search, reranker)
3. Click "Run Retriever Evaluation"

#### Prompt Evaluation:
1. Select prompt types to test
2. Select language models to test  
3. Optionally limit number of questions
4. Click "Run Prompt Evaluation"

#### Combined Evaluation:
1. Select all parameters for both retriever and prompt evaluation
2. Use presets for quick configuration
3. Click "Run Combined Evaluation"

### 4. View Results
- Results are automatically logged to Weights & Biases
- CSV files saved to `assets/csv/evaluation_results/`
- Access results history via the "View Results" section

## Evaluation Metrics

### Retriever Metrics:
- **MRR (Mean Reciprocal Rank)**: Average reciprocal rank of correct answers
- **Hit@K**: Percentage of queries where correct answer appears in top K results
- **Average Retrieval Time**: Time taken to retrieve results

### Prompt Metrics:
- **ROUGE-1/2/L**: Text overlap measures between generated and reference answers
- **Cosine Similarity**: Semantic similarity score
- **Quality Scores**: AI-evaluated scores for clarity, relevancy, and factual correctness

## Integration with Existing Code

The evaluation UI integrates with existing evaluation modules:

- `hubert.retriever_evaluate.evaluate_retrievers.RetrieverEvaluator`
- `hubert.prompt_evaluation.evaluators.wandb_evaluation.run_evaluation`
- `hubert.prompt_evaluation.prompts.prompt_templates.PromptFactory`

## Configuration

The system uses settings from `hubert.config.py`:

- `wandb_project`: Weights & Biases project name
- `wandb_entity`: Weights & Biases entity
- `embedding_model`: Default embedding model
- `reranker_model`: Default reranker model
- Other evaluation-specific settings

## Error Handling

- CSV file validation with helpful error messages
- Form validation for required fields
- Graceful handling of evaluation errors
- User-friendly flash messages for all operations

## File Structure

```
ui/app/evaluation/
├── __init__.py          # Module initialization
├── routes.py            # Main evaluation routes and logic
├── README.md           # This documentation
└── templates/
    ├── dashboard.html   # Main evaluation interface
    └── results.html     # Results viewing interface
```

## Dependencies

- Flask and Flask-Login for web framework and authentication
- pandas for CSV handling
- Existing HuBer evaluation modules
- Bootstrap 4 for UI components
- Font Awesome for icons 