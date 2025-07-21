#!/bin/bash
# Setup script for Trotro AI Service development environment

echo "🚀 Setting up Trotro AI Service..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models logs

echo "✅ Setup complete!"
echo ""
echo "To start the service:"
echo "1. source venv/bin/activate"
echo "2. python -m uvicorn app:app --reload"
echo ""
echo "API will be available at: http://localhost:8001"
echo "API docs will be available at: http://localhost:8001/docs"
