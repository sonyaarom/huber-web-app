#!/usr/bin/env bash
# exit on error
set -o errexit

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn eventlet flask-socketio --upgrade --no-cache-dir

# Verify gunicorn is installed and in path
which gunicorn || echo "gunicorn not found in path" 