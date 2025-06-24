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

# Configure logging properly for Flask app
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)
logger.info("=== VIEWS/MAIN MODULE LOADED ===")

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
@login_required
def chat():
    logger.info(f"User {current_user.username} accessed chat page")
    return render_template('index.html')

@bp.route('/search')
@login_required
def search():
    logger.info(f"User {current_user.username} accessed search page")
    return render_template('search.html')

@bp.route('/register', methods=['GET', 'POST'])
def register():
    try:
        if current_user.is_authenticated:
            logger.info(f"Already authenticated user {current_user.username} tried to access register")
            return redirect(url_for('main.chat'))
        
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            
            logger.info(f"=== REGISTRATION ATTEMPT ===")
            logger.info(f"Username: {username}")
            logger.info(f"Password provided: {'Yes' if password else 'No'}")
            logger.info(f"Password length: {len(password) if password else 0}")
            
            if not username or not password:
                logger.warning("Missing username or password")
                flash('Please provide both username and password.', 'danger')
                return redirect(url_for('main.register'))
            
            if len(password) < 4:
                logger.warning(f"Password too short for user: {username}")
                flash('Password must be at least 4 characters long.', 'danger')
                return redirect(url_for('main.register'))
            
            # Test database connection
            try:
                storage = PostgresStorage()
                logger.info("Database connection established for registration")
            except Exception as db_error:
                logger.error(f"Database connection failed during registration: {db_error}")
                flash('Database connection error. Please try again later.', 'danger')
                return redirect(url_for('main.register'))
            
            # Check if user already exists
            try:
                existing_user = storage.get_user_by_username(username)
                logger.info(f"Existing user check completed. User exists: {existing_user is not None}")
            except Exception as lookup_error:
                logger.error(f"Error checking existing user: {lookup_error}")
                flash('Registration error. Please try again later.', 'danger')
                return redirect(url_for('main.register'))
            
            if existing_user:
                logger.warning(f"Registration failed - username already exists: {username}")
                flash('Username already exists.', 'danger')
                return redirect(url_for('main.register'))
            
            # Hash password
            try:
                logger.info(f"Generating password hash for user: {username}")
                password_hash = generate_password_hash(password, method='pbkdf2:sha256')
                logger.info(f"Password hash generated successfully. Hash length: {len(password_hash)}")
                logger.info(f"Password hash starts with: {password_hash[:20]}...")
                
                # Test the hash immediately to verify it works
                test_verify = check_password_hash(password_hash, password)
                logger.info(f"Password hash verification test: {test_verify}")
                
                if not test_verify:
                    logger.error("Password hash verification test failed!")
                    flash('Password hashing error. Please try again.', 'danger')
                    return redirect(url_for('main.register'))
                    
            except Exception as hash_error:
                logger.error(f"Error generating password hash: {hash_error}")
                flash('Password hashing error. Please try again later.', 'danger')
                return redirect(url_for('main.register'))
            
            # Create user object
            try:
                new_user = User(
                    username=username,
                    password_hash=password_hash,
                    role='user'
                )
                logger.info(f"User object created: username={new_user.username}, role={new_user.role}")
                logger.info(f"User object password_hash length: {len(new_user.password_hash)}")
            except Exception as user_creation_error:
                logger.error(f"Error creating user object: {user_creation_error}")
                flash('User creation error. Please try again later.', 'danger')
                return redirect(url_for('main.register'))
            
            # Save to database
            try:
                user_id = storage.add_user(new_user)
                logger.info(f"User saved to database with ID: {user_id}")
                
                # Verify the user was saved correctly
                saved_user = storage.get_user_by_username(username)
                if saved_user:
                    logger.info(f"User verification successful: {saved_user.username}, role: {saved_user.role}")
                    logger.info(f"Saved password hash length: {len(saved_user.password_hash)}")
                    
                    # Test password verification on saved user
                    verification_test = check_password_hash(saved_user.password_hash, password)
                    logger.info(f"Password verification test on saved user: {verification_test}")
                else:
                    logger.error("User was not found after saving!")
                    flash('Registration error - user not saved properly.', 'danger')
                    return redirect(url_for('main.register'))
                    
            except Exception as save_error:
                logger.error(f"Error saving user to database: {save_error}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                flash('Database save error. Please try again later.', 'danger')
                return redirect(url_for('main.register'))
            
            logger.info(f"=== REGISTRATION SUCCESS for {username} ===")
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('main.login'))
        
        logger.info("Rendering registration page (GET request)")
        return render_template('register.html')
        
    except Exception as e:
        logger.error(f"Unexpected error in registration route: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        flash('An unexpected error occurred during registration. Please try again.', 'danger')
        return redirect(url_for('main.register'))

@bp.route('/login', methods=['GET', 'POST'])
def login():
    try:
        if current_user.is_authenticated:
            logger.info(f"Already authenticated user {current_user.username} tried to access login")
            return redirect(url_for('main.chat'))
        
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            remember = True if request.form.get('remember') else False
            
            logger.info(f"=== LOGIN ATTEMPT ===")
            logger.info(f"Username: {username}")
            logger.info(f"Password provided: {'Yes' if password else 'No'}")
            logger.info(f"Remember me: {remember}")
            
            if not username or not password:
                logger.warning("Missing username or password")
                flash('Please provide both username and password.', 'danger')
                return redirect(url_for('main.login'))
            
            # Test database connection
            try:
                storage = PostgresStorage()
                logger.info("Database connection established")
            except Exception as db_error:
                logger.error(f"Database connection failed: {db_error}")
                flash('Database connection error. Please try again later.', 'danger')
                return redirect(url_for('main.login'))
            
            # Look up user
            try:
                user = storage.get_user_by_username(username)
                logger.info(f"Database query completed. User found: {user is not None}")
            except Exception as lookup_error:
                logger.error(f"Error looking up user: {lookup_error}")
                flash('Login error. Please try again later.', 'danger')
                return redirect(url_for('main.login'))
            
            if user is None:
                logger.warning(f"User not found in database: {username}")
                flash('Please check your login details and try again.', 'danger')
                return redirect(url_for('main.login'))
            
            logger.info(f"User found: {user.username}, role: {user.role}, ID: {user.id}")
            logger.info(f"Stored password hash length: {len(user.password_hash)}")
            logger.info(f"Stored password hash preview: {user.password_hash[:50]}...")
            
            # Verify password
            try:
                logger.info(f"Attempting password verification for user: {username}")
                logger.info(f"Input password length: {len(password)}")
                
                password_valid = check_password_hash(user.password_hash, password)
                logger.info(f"Password verification result: {password_valid}")
                
                # Additional debugging - test with a known working hash
                if not password_valid:
                    logger.info("Password verification failed. Testing hash generation...")
                    test_hash = generate_password_hash(password, method='pbkdf2:sha256')
                    test_verify = check_password_hash(test_hash, password)
                    logger.info(f"Test hash verification (should be True): {test_verify}")
                    logger.info(f"Test hash preview: {test_hash[:50]}...")
                    
            except Exception as password_error:
                logger.error(f"Error verifying password: {password_error}")
                import traceback
                logger.error(f"Password verification traceback: {traceback.format_exc()}")
                flash('Login error. Please try again later.', 'danger')
                return redirect(url_for('main.login'))
            
            if not password_valid:
                logger.warning(f"Invalid password for user: {username}")
                flash('Please check your login details and try again.', 'danger')
                return redirect(url_for('main.login'))
            
            # Log in user
            try:
                logger.info(f"Attempting to log in user: {username}")
                login_user(user, remember=remember)
                logger.info(f"login_user() completed successfully")
                logger.info(f"current_user.is_authenticated: {current_user.is_authenticated}")
                logger.info(f"current_user.username: {getattr(current_user, 'username', 'N/A')}")
            except Exception as login_error:
                logger.error(f"Error during login_user(): {login_error}")
                flash('Login error. Please try again later.', 'danger')
                return redirect(url_for('main.login'))
            
            flash('Logged in successfully!', 'success')
            logger.info(f"=== LOGIN SUCCESS for {username} ===")
            return redirect(url_for('main.chat'))

        logger.info("Rendering login page (GET request)")
        return render_template('login.html')
        
    except Exception as e:
        logger.error(f"Unexpected error in login route: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        flash('An unexpected error occurred. Please try again.', 'danger')
        return redirect(url_for('main.login'))

@bp.route('/logout')
@login_required
def logout():
    username = current_user.username
    logout_user()
    logger.info(f"User {username} logged out successfully")
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.index'))

@bp.route('/evaluation')
@login_required
def evaluation_dashboard():
    if not current_user.is_admin:
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('main.chat'))
    return render_template('evaluation.html')

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
    if not current_user.is_admin:
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
