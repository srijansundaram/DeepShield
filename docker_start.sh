#!/bin/bash
# DeepShield startup — runs Streamlit + FastAPI in parallel

echo "🛡️  DeepShield starting..."

# Check checkpoints
if ls /app/checkpoints/*.pth 1>/dev/null 2>&1; then
    echo "✓ Checkpoints found"
else
    echo "⚠ No checkpoints in /app/checkpoints/ — app will run in demo mode"
    echo "  Mount your checkpoints: -v /path/to/checkpoints:/app/checkpoints"
fi

# Start FastAPI in background
echo "Starting REST API on port 8000..."
uvicorn api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Start Streamlit in foreground
echo "Starting Streamlit on port 8501..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# If streamlit exits, kill API too
kill $API_PID