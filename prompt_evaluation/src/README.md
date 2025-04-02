

prompt_evaluation/
├── src/
│ ├── core/
│ │ ├── prompt_builder.py # Core prompt building functionality
│ │ ├── prompt_evaluation.py # Main evaluation logic
│ │ └── main_prompt_evaluation.py
│ ├── prompts/
│ │ ├── simple_prompt.py # Basic prompt implementation
│ │ ├── medium_prompt.py # Enhanced prompt with style guidelines
│ │ └── advanced_prompt.py # Advanced prompt with additional features
│ └── utils/
│ ├── rag_helpers.py # RAG utility functions
│ └── retriever.py # Document retrieval implementation
└── config/
└── settings.py # Configuration settings

## Features

- **Flexible Prompt Building**: Modular system for creating and testing different prompt strategies
- **Three Complexity Levels**:
  - Simple: Basic RAG prompts
  - Medium: Enhanced with style guidelines
  - Advanced: Includes examples, quality checks, and additional context
- **Performance Metrics**:
  - Retrieval time
  - Generation time
  - Total processing time
  - Success/failure rates

## Prerequisites

- Python 3.8+
- Langfuse account
- Pinecone account
- Required environment variables (see Configuration section)

## Installation
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir
1. Clone the repository:

```bash
git clone [repository-url]
cd prompt-evaluation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your credentials
```

## Configuration

Required environment variables:
- `LANGFUSE_SECRET_KEY`: Your Langfuse secret key
- `LANGFUSE_PUBLIC_KEY`: Your Langfuse public key
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_ENVIRONMENT`: Pinecone environment
- `PINECONE_INDEX_NAME`: Name of your Pinecone index
- `MODEL_PATH`: Path to your LLM model
- Additional configuration options in `config/settings.py`

## Usage

1. Basic prompt evaluation:
```python
from src.prompts.simple_prompt import SimplePrompt

simple_prompt = SimplePrompt()
result = simple_prompt.build_prompt("Your question", "Your context")
```

2. Run complete evaluation:
```python
python -m src.core.prompt_evaluation
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request



git clone https://github.com/abetlen/llama-cpp-python.git && cd llama-cpp-python && CUDA_HOME=/usr/local/cuda CUDA_DOCKER_ARCH=all CMAKE_ARGS="-DGGML_CUDA=ON" pip install .


