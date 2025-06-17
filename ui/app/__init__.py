from flask import Flask
from flask_socketio import SocketIO
from hubert.config import settings

socketio = SocketIO()

def create_app():
    app = Flask(__name__, static_folder='static')
    app.config["SECRET_KEY"] = settings.secret_key

    from .views.main import bp as main_blueprint
    app.register_blueprint(main_blueprint)

    socketio.init_app(app)
    return app