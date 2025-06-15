# import sys
# import os

# # Add the project root to the Python path
# sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# # Import and run the app
# from ui.app import app, socketio

# if __name__ == '__main__':
#     socketio.run(app, host='0.0.0.0', port=1234, debug=True)

# import sys
# import os

# # Add the project root to the Python path
# sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# # Import and run the app
# from ui.app import app, socketio

# if __name__ == '__main__':
#     # For local development
#     socketio.run(app, host='0.0.0.0', port=1234, debug=True)
# else:
#     # For production
#     # No need to call socketio.run() as gunicorn will handle this
#     pass


import os
import sys
from ui.app import create_app, socketio

# Ensure the root directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 1234))
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)