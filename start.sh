#!/bin/bash
set -e

echo "=== KTC Predictor Startup ==="
echo "Python version: $(python --version)"
echo "Node version: $(node --version)"

# Start backend
echo ""
echo "=== Starting Backend ==="
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 5000 &
BACKEND_PID=$!
cd ..

# Wait for backend to be ready (up to 60 seconds)
echo "Waiting for backend..."
for i in {1..60}; do
    if curl -s http://localhost:5000/health > /dev/null 2>&1; then
        echo "Backend is ready! (took ${i}s)"
        break
    fi
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "ERROR: Backend process died!"
        exit 1
    fi
    sleep 1
done

# Final check
if ! curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "ERROR: Backend failed to start after 60 seconds"
    exit 1
fi

# Start frontend
echo ""
echo "=== Starting Frontend on port $PORT ==="
cd frontend
exec npm start
