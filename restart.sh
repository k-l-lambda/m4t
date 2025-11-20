#!/bin/bash
# M4T Server Restart Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== M4T Server Restart ==="

# Step 1: Stop existing server
echo "Step 1: Stopping existing m4t server..."
pkill -f "python.*server.py" 2>/dev/null || echo "No existing server found"
sleep 2

# Step 2: Clear Python cache
echo "Step 2: Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
echo "Cache cleared"

# Step 3: Start server
echo "Step 3: Starting m4t server..."
nohup ./env/bin/python server.py > /tmp/m4t_server.log 2>&1 &
SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"

# Step 4: Wait for server to be ready
echo "Step 4: Waiting for server to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "ERROR: Server did not start within 30 seconds"
        echo "Check log: tail -50 /tmp/m4t_server.log"
        exit 1
    fi
    sleep 1
    echo -n "."
done

# Step 5: Health check
echo ""
echo "Step 5: Health check..."
HEALTH=$(curl -s http://localhost:8000/health)
echo "$HEALTH" | python3 -m json.tool

# Check GPT-SoVITS availability
if curl -s http://localhost:9880/control > /dev/null 2>&1; then
    echo ""
    echo "✓ GPT-SoVITS service is running on port 9880"
else
    echo ""
    echo "⚠ WARNING: GPT-SoVITS service is NOT running on port 9880"
    echo "Start it with: cd /home/camus/work/GPT-SoVITS && ./env/bin/python api.py -dr reference.wav -dt 'reference text' -dl en"
fi

echo ""
echo "=== M4T Server Ready ==="
echo "Server PID: $SERVER_PID"
echo "Log file: /tmp/m4t_server.log"
echo "API docs: http://localhost:8000/docs"
echo ""
echo "Configuration:"
echo "  - Default port: 8000 (customize in .env.local)"
echo "  - See .env.local for all configuration options"
