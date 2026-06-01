#!/bin/bash
# Setup and Run Q-A-C Dataset Creation Pipeline
# This script sets up the environment and runs the dataset creation

set -e  # Exit on error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "================================================================================"
echo "Q-A-C Dataset Creation - Setup and Execution"
echo "================================================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.9+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python version: $(python3 --version)"

# Check/create virtual environment
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Install required packages
echo ""
echo "Installing required packages..."
pip install --quiet --upgrade pip
pip install --quiet datasets pandas numpy

# Check if other requirements are needed
if [ -f "requirements.txt" ]; then
    echo "Installing project requirements..."
    pip install --quiet -r requirements.txt
fi

echo ""
echo "✅ All packages installed"
echo ""

# Create data directory if it doesn't exist
mkdir -p data

# Run the pipeline
echo "================================================================================"
echo "Running Q-A-C Dataset Creation Pipeline"
echo "================================================================================"
echo ""

python3 examples/run_qac_creation_pipeline.py

echo ""
echo "================================================================================"
echo "Setup and Execution Complete!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Review the created dataset: data/qac_dataset_constitutional.json"
echo "  2. Validate: python3 examples/validate_and_enhance_qac_dataset.py --dataset data/qac_dataset_constitutional.json"
echo "  3. See documentation: docs/QAC_DATASET_CREATION_GUIDE.md"
echo ""
