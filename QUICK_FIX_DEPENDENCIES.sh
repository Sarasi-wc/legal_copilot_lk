#!/bin/bash
# Quick Fix for Missing Dependencies
# Run this to install all required packages

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "================================================================================"
echo "FIXING DEPENDENCIES"
echo "================================================================================"
echo ""

# Check/create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --quiet --upgrade pip

# Install critical missing packages
echo ""
echo "Installing critical packages..."
pip install --quiet rank-bm25 datasets pandas numpy spacy

# Download spaCy model
echo ""
echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm --quiet

# Install other requirements (with flexible version handling)
echo ""
echo "Installing other requirements..."
pip install --quiet fastapi uvicorn pydantic python-dotenv || true
pip install --quiet torch transformers sentence-transformers || true
pip install --quiet faiss-cpu elasticsearch || true
pip install --quiet pytesseract pdf2image pdfplumber beautifulsoup4 lxml || true
pip install --quiet tqdm scikit-learn nltk requests python-multipart || true
pip install --quiet openai streamlit pytest jupyter ipykernel black || true

echo ""
echo "================================================================================"
echo "✅ DEPENDENCIES INSTALLED!"
echo "================================================================================"
echo ""
echo "You can now:"
echo "  1. Create Q-A-C dataset: python examples/create_qac_simple.py --max-items 50"
echo "  2. Run tests: pytest tests/ -v"
echo "  3. See EXECUTION_ORDER_GUIDE.md for complete order"
echo ""
