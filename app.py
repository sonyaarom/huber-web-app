from flask import Flask, render_template, request, jsonify
from flask_bootstrap import Bootstrap
import logging
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.generator.generator_utils.generator_utils import initialize_models, generate_answer
from src.generator.prompt_utils.prompt_templates import PromptFactory
from src.generator.prompt_utils.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='ui/app/templates')
app.config['SECRET_KEY'] = 'your-secret-key'
Bootstrap(app)

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
        response = generate_answer(
            llm=llm,
            generation_type='openai',
            prompt_text=prompt_text,
            max_tokens=256,
            temperature=0.1
        )
        
        # Extract the answer text
        answer = response['choices'][0]['text']
        
        return jsonify({
            'question': question,
            'context': context,
            'answer': answer
        })
    
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 