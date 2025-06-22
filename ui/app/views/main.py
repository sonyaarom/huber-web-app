from flask import Blueprint, render_template, request, flash, redirect, url_for, session, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from hubert.db.models import User
from hubert.db.postgres_storage import PostgresStorage
from hubert.config import settings, reload_settings
from hubert.main import rag_main_func, retrieve_urls as get_urls, reinitialize_retriever
from hubert.common.utils.ner_utils import extract_entities
from flask_socketio import emit
import sys
import os
import logging
from dotenv import set_key, find_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

# Import the RAG function and NER utility
from hubert.main import rag_main_func, retrieve_urls as get_urls
from hubert.common.utils.ner_utils import extract_entities


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

@bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.chat'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        storage = PostgresStorage()
        user = storage.get_user_by_username(username)
        
        if user:
            flash('Username already exists.', 'danger')
            return redirect(url_for('main.register'))
        
        new_user = User(
            username=username,
            password_hash=generate_password_hash(password, method='pbkdf2:sha256'),
            role='user'
        )
        storage.add_user(new_user)
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('main.login'))
        
    return render_template('register.html')

@bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.chat'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False
        
        storage = PostgresStorage()
        user = storage.get_user_by_username(username)
        
        if not user or not check_password_hash(user.password_hash, password):
            flash('Please check your login details and try again.', 'danger')
            return redirect(url_for('main.login'))
            
        login_user(user, remember=remember)
        return redirect(url_for('main.chat'))

    return render_template('login.html')

@bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.index'))

# SocketIO event handlers will be registered in the main app
def handle_message(message):
    logger.info(f"Received message: {message}")
    try:
        ner_filters = None
        if settings.use_ner:
            ner_filters = extract_entities(message)
            logger.info(f"Extracted NER filters: {ner_filters}")

        # Get both the answer and the URLs from the RAG function
        response_data = rag_main_func(message, ner_filters=ner_filters)
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
        response_data = get_urls(question)
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error retrieving URLs: {e}")
        return jsonify({"error": str(e)}), 500
    

@bp.route('/config', methods=['GET', 'POST'])
@login_required
def config():
    """
    Route to display and update application configuration.
    """
    if not current_user.is_admin():
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('main.chat'))
        
    # Define the env file name and path
    env_file = '.venv'
    dotenv_path = os.path.join(settings.base_dir, env_file)
    
    # Create the file if it doesn't exist
    if not os.path.exists(dotenv_path):
        with open(dotenv_path, 'w') as f:
            pass

    # Always reload settings first to get fresh values
    current_settings = reload_settings()

    if request.method == 'POST':
        try:
            # Update settings from form
            form_settings = {
                "EMBEDDING_MODEL": request.form.get('embedding_model'),
                "RERANKER_MODEL": request.form.get('reranker_model'),
                "HYBRID_ALPHA": request.form.get('hybrid_alpha'),
                "USE_NER": 'True' if request.form.get('use_ner') == 'on' else 'False',
                "USE_RERANKER": 'True' if request.form.get('use_reranker') == 'on' else 'False'
            }
            
            # Update each setting in the .venv file
            for key, value in form_settings.items():
                set_key(dotenv_path, key, value)
            
            # Reload settings and reinitialize retriever
            current_settings = reload_settings()
            reinitialize_retriever()
            
            flash('Configuration updated successfully!', 'success')
        except Exception as e:
            flash(f'Error updating configuration: {e}', 'danger')

        return redirect(url_for('main.config'))

    # For both GET and POST, use the freshly reloaded settings
    return render_template('config.html', title='Configuration', config=current_settings)
