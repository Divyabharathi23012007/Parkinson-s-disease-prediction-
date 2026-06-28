#!/bin/bash
# NeuroScan Parkinson's AI Platform — Start Script

echo ""
echo "╔══════════════════════════════════════╗"
echo "║   NeuroScan — Parkinson's AI System  ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required. Please install it first."
    exit 1
fi

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r backend/requirements.txt -q

echo ""
echo "🚀 Starting Flask API server on http://localhost:5000 ..."
echo "🌐 Open frontend/index.html in your browser"
echo ""
echo "Press Ctrl+C to stop"
echo ""

cd backend && python3 app.py
