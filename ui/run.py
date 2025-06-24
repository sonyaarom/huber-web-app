import sys
import os
import logging

# Configure logging FIRST before importing anything else
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

logger = logging.getLogger(__name__)
logger.info("=== STARTING FLASK APPLICATION ===")

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from ui.app import app, socketio

if __name__ == '__main__':
    logger.info("Starting SocketIO server on host=0.0.0.0, port=1234, debug=True")
    socketio.run(app, host='0.0.0.0', port=1234, debug=True)