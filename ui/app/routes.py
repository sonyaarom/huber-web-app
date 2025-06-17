from flask import render_template, request, jsonify
from ui.app import app
from src.generator.generator_utils.generator_utils import initialize_models, generate_answer
from src.generator.main import together_generator
from src.generator.prompt_utils.prompt_templates import PromptFactory
from src.generator.prompt_utils.config import settings
from hubert.main import retrieve_urls as retrieve_urls_main
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models globally to avoid reloading them on each request
try:
    llm, embedding_generator, reranker_model = initialize_models(model_type="openai")
    prompt_factory = PromptFactory().create_prompt(prompt_type=settings.DEFAULT_PROMPT_TYPE)
    logger.info("Models initialized successfully")
except Exception as e:
    logger.error(f"Error initializing models: {e}")
    llm, embedding_generator, reranker_model, prompt_factory = None, None, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    # Get the question from the form
    question = request.form.get('question', '')
    context = request.form.get('context', '')
    
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    
    try:
        # Check if models are initialized
        if llm is None or prompt_factory is None:
            return jsonify({'error': 'Models not initialized properly'}), 500
        
        # Build the prompt
        prompt_text = prompt_factory.build_prompt(user_question=question, context=context)
        
        # Generate the answer
        response = together_generator(question, context)
        # Extract the answer text
        
        return jsonify({
            'question': question,
            'context': context,
            'answer': response
        })
    
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/search')
def search():
    return render_template('search.html')

@app.route('/retrieve_urls', methods=['POST'])
def retrieve_urls_endpoint():
    question = request.form.get('question', '')
    
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    
    try:
        results = retrieve_urls_main(question)
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Error retrieving URLs: {e}")
        return jsonify({'error': str(e)}), 500