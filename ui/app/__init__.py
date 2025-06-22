from flask import Flask
from flask_login import LoginManager
from flask_socketio import SocketIO
from ui.app.views.main import bp as main_blueprint
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

    # Initialize extensions
    login_manager.init_app(app)
    socketio.init_app(app, cors_allowed_origins="*")

    # Register blueprints
    app.register_blueprint(main_blueprint)

    # Register socketio event handlers - import here to avoid circular imports
    from ui.app.views.main import handle_message
    socketio.on_event('message', handle_message)

    return app
