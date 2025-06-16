from flask import Blueprint, render_template, request, jsonify
from flask_socketio import emit
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
from hubert.main import rag_main_func, retrieve_urls as get_urls

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
    logger.info(f"Received message: {message}")
    try:
        # Get both the answer and the URLs from the RAG function
        response_data = rag_main_func(message)
        logger.info(f"RAG function returned: {type(response_data)} - {response_data}")
        
        # Check if response_data is in the expected format
        if isinstance(response_data, dict) and 'answer' in response_data:
            # Emit a single event with both answer and URLs
            emit('response', response_data)
            logger.info(f"Sending response: {response_data['answer'][:100]}...")
        elif isinstance(response_data, str):
            # If it's just a string, emit it as a message
            emit('message', response_data)
            logger.info(f"Sending message: {response_data[:100]}...")
        else:
            # Fallback: convert to string and send as message
            response_str = str(response_data)
            emit('message', response_str)
            logger.info(f"Sending converted message: {response_str[:100]}...")
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        emit('error', {'error': str(e)})

@bp.route('/retrieve_urls', methods=['POST'])
def retrieve_urls_endpoint():
    try:
        question = request.form.get('question')
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        # This now uses the cached retriever
        response_data = rag_main_func(question)
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error retrieving URLs: {e}")
        return jsonify({"error": str(e)}), 500