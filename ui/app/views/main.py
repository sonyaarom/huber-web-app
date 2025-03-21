from flask import Blueprint, render_template, request, jsonify
from ui.app import socketio
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

# Import the RAG function
from src.main import rag_main_func, retrieve_urls as get_urls

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    return render_template('landing.html')

@bp.route('/chat')
def chat():
    return render_template('index.html')

@bp.route('/search')
def search():
    return render_template('search.html')

@socketio.on('message')
def handle_message(message):
    try:
        logger.info(f"Received message: {message}")
        
        # Use the RAG function to get a response
        response = rag_main_func(message)
        
        # Extract the answer text from the response
        if isinstance(response, dict) and 'choices' in response:
            answer = response['choices'][0]['text']
        else:
            answer = str(response)
        
        logger.info(f"Sending response: {answer[:100]}...")
        socketio.send(answer)
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        socketio.send(f"I'm sorry, I encountered an error: {str(e)}")

@bp.route('/retrieve_urls', methods=['POST'])
def retrieve_urls_endpoint():
    question = request.form.get('question', '')
    urls = get_urls(question)
from flask import Blueprint, render_template, request, jsonify
from ui.app import socketio
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

# Import the RAG function
from src.main import rag_main_func, retrieve_urls as get_urls

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    return render_template('landing.html')

@bp.route('/chat')
def chat():
    return render_template('index.html')

@bp.route('/search')
def search():
    return render_template('search.html')

@socketio.on('message')
def handle_message(message):
    try:
        logger.info(f"Received message: {message}")
        
        # Use the RAG function to get a response
        response = rag_main_func(message)
        
        # Extract the answer text from the response
        if isinstance(response, dict) and 'choices' in response:
            answer = response['choices'][0]['text']
        else:
            answer = str(response)
        
        logger.info(f"Sending response: {answer[:100]}...")
        socketio.send(answer)
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        socketio.send(f"I'm sorry, I encountered an error: {str(e)}")

@bp.route('/retrieve_urls', methods=['POST'])
def retrieve_urls_endpoint():
    question = request.form.get('question', '')
    urls = get_urls(question)
    return jsonify({'urls': urls})