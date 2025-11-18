#!/bin/bash
# Start SeamlessM4T API server in development mode

echo "Starting SeamlessM4T API Server (Development Mode)"
echo "=================================================="

# Set proxy for model downloads
export HTTP_PROXY=http://localhost:1091
export HTTPS_PROXY=http://localhost:1091

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Start server
echo ""
echo "Starting FastAPI server..."
echo "API Documentation: http://localhost:8000/docs"
echo "Press Ctrl+C to stop"
echo ""

python server.py
