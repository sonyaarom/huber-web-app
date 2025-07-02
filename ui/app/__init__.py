from flask import Flask
from flask_login import LoginManager
from flask_socketio import SocketIO
import logging

# Configure logging at the top level - this should be one of the first things that happens
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Force reconfiguration
)

# Get logger for this module
logger = logging.getLogger(__name__)
logger.info("=== FLASK APP INITIALIZATION START ===")

# Initialize Sentry as early as possible
from hubert.common.monitoring import init_sentry
sentry_initialized = init_sentry()
if sentry_initialized:
    logger.info("✅ Sentry APM initialized successfully")
else:
    logger.info("⚠️  Sentry APM not initialized (DSN not configured)")

from ui.app.views.main import bp as main_blueprint
from ui.app.evaluation import bp as evaluation_blueprint
from hubert.db.models import User
from hubert.db.postgres_storage import PostgresStorage

# Initialize LoginManager
login_manager = LoginManager()
login_manager.login_view = 'main.login' # The route for the login page
login_manager.login_message_category = 'info'

# Initialize SocketIO
socketio = SocketIO()

@login_manager.user_loader
def load_user(user_id):
    """Loads a user from the database for Flask-Login."""
    storage = PostgresStorage()
    return storage.get_user_by_id(int(user_id))

def create_app():
    """Creates and configures the Flask application."""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'a-very-secret-key' # IMPORTANT: Change this in a real application

    # Configure Flask's own logger to match our format
    app.logger.setLevel(logging.INFO)
    
    logger.info("=== CONFIGURING FLASK APP ===")

    # Initialize extensions
    login_manager.init_app(app)
    socketio.init_app(app, cors_allowed_origins="*")

    # Register blueprints
    app.register_blueprint(main_blueprint)
    app.register_blueprint(evaluation_blueprint)

    # Register socketio event handlers - import here to avoid circular imports
    from ui.app.views.main import handle_message
    socketio.on_event('message', handle_message)

    logger.info("=== FLASK APP CONFIGURATION COMPLETE ===")
    return app

# Create the app instance
app = create_app()

@app.route("/health")
def health():
    return "ok", 200
