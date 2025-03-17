from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__, static_folder='static')
socketio = SocketIO(app)

from ui.app.views.main import bp as main_bp
app.register_blueprint(main_bp)