#!/bin/bash
# Quick Environment Setup Script
# Creates .env file with OpenAI API key prompt

echo "================================================================================"
echo "Environment Variables Setup"
echo "================================================================================"
echo ""

# Check if .env exists
if [ -f ".env" ]; then
    echo "⚠️  .env file already exists"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing .env file"
        exit 0
    fi
fi

# Copy example
if [ -f ".env.example" ]; then
    cp .env.example .env
    echo "✅ Copied .env.example to .env"
else
    echo "⚠️  .env.example not found, creating basic .env"
    touch .env
fi

echo ""
echo "Please enter your OpenAI API key:"
echo "(Get it from: https://platform.openai.com/api-keys)"
echo ""
read -p "OpenAI API Key: " api_key

if [ -z "$api_key" ]; then
    echo "⚠️  No API key provided. You can add it later to .env file"
else
    # Update .env file
    if grep -q "OPENAI_API_KEY=" .env; then
        # Replace existing
        sed -i "s|OPENAI_API_KEY=.*|OPENAI_API_KEY=$api_key|" .env
    else
        # Add new
        echo "OPENAI_API_KEY=$api_key" >> .env
    fi
    echo "✅ API key added to .env"
fi

echo ""
echo "================================================================================"
echo "✅ Setup Complete!"
echo "================================================================================"
echo ""
echo "📝 Your .env file is ready"
echo "📖 See docs/ENVIRONMENT_VARIABLES_GUIDE.md for all options"
echo ""
