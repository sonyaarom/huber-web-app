import json
import logging
from functools import wraps
from flask import render_template, request, jsonify, flash, redirect, url_for, session, abort
from flask_login import login_user, logout_user, current_user, login_required
from werkzeug.security import check_password_hash, generate_password_hash
from ui.app import app # Import the app instance
from hubert.db.postgres_storage import PostgresStorage
from hubert.db.models import User
from hubert.retriever_evaluate.evaluate_retrievers import run_evaluation as run_retriever_evaluation_script

# Your new imports and initializations
# Note: Some paths might need adjustment from 'src' to 'hubert' depending on your final structure
from hubert.generator.generator_utils.generator_utils import initialize_models
from hubert.generator.main import together_generator
from hubert.generator.prompt_utils.prompt_templates import PromptFactory
from hubert.generator.prompt_utils.config import settings
from hubert.main import retrieve_urls as retrieve_urls_main

# Configure logging properly for Flask app
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # This forces reconfiguration even if basicConfig was called elsewhere
)

# Get the logger for this module
logger = logging.getLogger(__name__)

# Also configure the app logger
app.logger.setLevel(logging.INFO)

# Ensure our logger messages appear
logger.info("=== ROUTES MODULE LOADED ===")

# Initialize models globally
try:
    llm, embedding_generator, reranker_model = initialize_models(model_type="openai")
    prompt_factory = PromptFactory().create_prompt(prompt_type=settings.DEFAULT_PROMPT_TYPE)
    logger.info("Models initialized successfully")
except Exception as e:
    logger.error(f"Error initializing models: {e}")
    llm, embedding_generator, reranker_model, prompt_factory = None, None, None, None

# +-----------------------------------+
# |  ADMIN REQUIRED DECORATOR         |
# +-----------------------------------+
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            abort(403) # Forbidden
        return f(*args, **kwargs)
    return decorated_function

# +-----------------------------------+
# |  AUTHENTICATION ROUTES            |
# +-----------------------------------+
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        logger.info(f"Login attempt for username: {username}")
        
        storage = PostgresStorage()
        user_data = storage.get_user_by_username(username)
        
        if user_data is None:
            logger.warning(f"User not found: {username}")
            flash('Invalid username or password', 'danger')
            return redirect(url_for('login'))
        
        logger.info(f"User found: {user_data.username}, role: {user_data.role}")
        
        if not check_password_hash(user_data.password_hash, password):
            logger.warning(f"Invalid password for user: {username}")
            flash('Invalid username or password', 'danger')
            return redirect(url_for('login'))
        
        logger.info(f"Password verified for user: {username}")
        login_user(user_data, remember=True)
        flash('Logged in successfully!', 'success')
        return redirect(url_for('index'))
    return render_template('login.html', title='Sign In')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        storage = PostgresStorage()
        if storage.get_user_by_username(username):
            flash('Username already exists.', 'warning')
            return redirect(url_for('register'))
        
        # Create user with hashed password
        password_hash = generate_password_hash(password)
        user = User(username=username, password_hash=password_hash, role='user')
        storage.add_user(user)
        
        flash('Congratulations, you are now a registered user!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register')

# +-----------------------------------+
# |  APPLICATION & USER ROUTES        |
# +-----------------------------------+
@app.route('/')
def index():
    return render_template('landing.html') # Changed to landing.html to match previous structure

@app.route('/chat')
@login_required
def chat():
    return render_template('chat.html')

@app.route('/search')
@login_required
def search():
    return render_template('search.html')

@app.route('/ask', methods=['POST'])
@login_required
def ask():
    question = request.form.get('question', '')
    context = request.form.get('context', '')
    if not question: return jsonify({'error': 'Question is required'}), 400
    try:
        if llm is None or prompt_factory is None: return jsonify({'error': 'Models not initialized properly'}), 500
        response = together_generator(question, context)
        return jsonify({'question': question, 'context': context, 'answer': response})
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/retrieve_urls', methods=['POST'])
@login_required
def retrieve_urls_endpoint():
    question = request.form.get('question', '')
    if not question: return jsonify({'error': 'Question is required'}), 400
    try:
        results = retrieve_urls_main(question)
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error retrieving URLs: {e}")
        return jsonify({'error': str(e)}), 500

# +-----------------------------------+
# |  ADMIN-ONLY ROUTES                |
# +-----------------------------------+
@app.route('/config', methods=['GET', 'POST'])
@login_required
@admin_required
def config():
    config_path = 'hubert/config.json'
    if request.method == 'POST':
        with open(config_path, 'r') as f: config_data = json.load(f)
        for key, value in request.form.items():
            if key in config_data:
                if isinstance(config_data[key], int): config_data[key] = int(value)
                elif isinstance(config_data[key], float): config_data[key] = float(value)
                elif isinstance(config_data[key], bool): config_data[key] = value.lower() in ['true', '1', 't']
                else: config_data[key] = value
        with open(config_path, 'w') as f: json.dump(config_data, f, indent=4)
        flash('Configuration updated successfully!', 'success')
        return redirect(url_for('config'))
    with open(config_path, 'r') as f: config_data = json.load(f)
    return render_template('config.html', title='Configuration', config=config_data)

@app.route('/evaluation')
@login_required
@admin_required
def evaluation_dashboard():
    try:
        storage = PostgresStorage()
        embedding_tables = storage.get_embedding_tables()
        return render_template('evaluation.html', title='Evaluation', embedding_tables=embedding_tables)
    except Exception as e:
        flash(f"Error loading evaluation page: {e}", 'danger')
        return render_template('evaluation.html', title='Evaluation', embedding_tables=[])

@app.route('/run_retriever_evaluation', methods=['POST'])
@login_required
@admin_required
def run_retriever_evaluation():
    try:
        selected_tables = request.form.getlist('embedding_tables')
        if not selected_tables:
            flash('Please select at least one embedding table to evaluate.', 'warning')
            return redirect(url_for('evaluation_dashboard'))
        results_df = run_retriever_evaluation_script(selected_tables)
        results_html = results_df.to_html(classes='table table-striped', index=False)
        return render_template('evaluation_results.html', title='Retriever Evaluation Results', results_html=results_html)
    except Exception as e:
        flash(f"An error occurred during evaluation: {e}", 'danger')
        return redirect(url_for('evaluation_dashboard'))
