#!/bin/bash

set -e  # Exit on any error

# Configuration
ENVIRONMENT=${1:-production}
REPO_URL="https://github.com/sonyaarom/data-engineering-huber.git"
APP_DIR="/opt/huber"
BACKUP_DIR="/opt/backups/huber-$(date +%Y%m%d-%H%M%S)"

echo "=== HuBer Deployment Script ==="
echo "Environment: $ENVIRONMENT"
echo "Target directory: $APP_DIR"
echo "Backup directory: $BACKUP_DIR"
echo ""

# Check if environment file exists locally
if [[ ! -f .venv.production ]]; then
    echo "❌ Error: .venv.production file not found!"
    echo "This file should be created by GitHub Actions."
    exit 1
fi

# SSH connection details from environment or deployment/.env
if [[ -f deployment/.env ]]; then
    source deployment/.env
fi

HETZNER_HOST=${HETZNER_HOST:-}
HETZNER_USER=${HETZNER_USER:-root}
HETZNER_SSH_KEY=${HETZNER_SSH_KEY:-/tmp/ssh_key}

if [[ -z "$HETZNER_HOST" ]]; then
    echo "❌ Error: HETZNER_HOST not set!"
    exit 1
fi

echo "🔗 Connecting to $HETZNER_USER@$HETZNER_HOST"

# Function to run commands on remote server
run_remote() {
    ssh -o ConnectTimeout=30 \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -i "$HETZNER_SSH_KEY" \
        "$HETZNER_USER@$HETZNER_HOST" "$@"
}

# Function to copy files to remote server
copy_to_remote() {
    scp -o ConnectTimeout=30 \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -i "$HETZNER_SSH_KEY" \
        "$1" "$HETZNER_USER@$HETZNER_HOST:$2"
}

echo "📋 Preparing deployment..."

# Test SSH connection
echo "🔌 Testing SSH connection..."
if ! run_remote "echo 'SSH connection successful'"; then
    echo "❌ Failed to connect to server via SSH"
    exit 1
fi

# Check if Docker is installed on the server
echo "🐳 Checking Docker installation..."
if ! run_remote "command -v docker >/dev/null 2>&1"; then
    echo "❌ Docker is not installed on the server!"
    echo "Please install Docker and docker-compose on your Hetzner server first."
    exit 1
fi

if ! run_remote "command -v docker-compose >/dev/null 2>&1 || command -v docker compose >/dev/null 2>&1"; then
    echo "❌ docker-compose is not installed on the server!"
    echo "Please install docker-compose on your Hetzner server first."
    exit 1
fi

# Create backup of current deployment if it exists
echo "💾 Creating backup of current deployment..."
run_remote "
    if [[ -d '$APP_DIR' ]]; then
        sudo mkdir -p '$BACKUP_DIR'
        sudo cp -r '$APP_DIR' '$BACKUP_DIR/'
        echo 'Backup created at $BACKUP_DIR'
    else
        echo 'No existing deployment found, skipping backup'
    fi
"

# Prepare the application directory
echo "📁 Preparing application directory..."
run_remote "
    sudo mkdir -p '$APP_DIR'
    sudo chown -R ${HETZNER_USER:-root}:${HETZNER_USER:-root} '$APP_DIR'
"

# Clone or update the repository
echo "📥 Updating application code..."
run_remote "
    cd '$APP_DIR'
    
    if [[ -d .git ]]; then
        echo 'Updating existing repository...'
        git fetch origin
        git reset --hard origin/main
        git clean -fd
    else
        echo 'Cloning repository...'
        git clone '$REPO_URL' .
    fi
    
    # Stop and remove old containers before building new ones
    if [[ -f docker-compose.yml ]]; then
        echo 'Ensuring all old containers are down before build...'
        docker-compose down --remove-orphans || docker compose down --remove-orphans || true
    fi
    
    echo 'Current commit:'
    git log -1 --oneline
"

# Copy environment file to server
echo "⚙️  Uploading environment configuration..."
copy_to_remote ".venv.production" "$APP_DIR/.venv"

# Build and start the application
echo "🚀 Building and starting the application..."
run_remote "
    cd '$APP_DIR'
    
    # Pull the latest base images
    docker-compose pull || docker compose pull || true
    
    # Build the application
    echo 'Building Docker image...'
    docker-compose build || docker compose build
    
    # Start the application
    echo 'Starting the application...'
    docker-compose up -d || docker compose up -d
    
    # Show container status
    echo 'Container status:'
    docker-compose ps || docker compose ps
"

# Wait for application to start
echo "⏳ Waiting for application to start..."
sleep 30

# Health check
echo "🏥 Performing health check..."
max_attempts=10
attempt=1

while [[ $attempt -le $max_attempts ]]; do
    if run_remote "curl -f -s http://localhost:1234/health >/dev/null 2>&1"; then
        echo "✅ Health check passed on attempt $attempt"
        break
    else
        echo "⏳ Health check failed, attempt $attempt/$max_attempts"
        if [[ $attempt -eq $max_attempts ]]; then
            echo "❌ Health check failed after $max_attempts attempts"
            echo "🔍 Checking container logs:"
            run_remote "cd '$APP_DIR' && docker-compose logs --tail=50"
            exit 1
        fi
        sleep 10
    fi
    attempt=$((attempt + 1))
done

# Show final status
echo ""
echo "🎉 Deployment completed successfully!"
echo ""
echo "📊 Final status:"
run_remote "
    cd '$APP_DIR'
    echo '--- Container Status ---'
    docker-compose ps || docker compose ps
    echo ''
    echo '--- Disk Usage ---'
    df -h '$APP_DIR'
    echo ''
    echo '--- Application Logs (last 10 lines) ---'
    docker-compose logs --tail=10 || docker compose logs --tail=10
"

echo ""
echo "🌐 Application should be available at: http://$HETZNER_HOST:1234"
echo "📝 Backup created at: $BACKUP_DIR (on server)"
echo ""
echo "✨ Deployment completed successfully!" 