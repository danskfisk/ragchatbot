#!/usr/bin/env bash

# Code formatting and quality checks script

echo "🔧 Running code quality checks..."

# Run isort to sort imports
echo "📦 Sorting imports with isort..."
uv run isort backend/ --check-only

if [ $? -ne 0 ]; then
    echo "❌ Import sorting issues found. Run 'uv run isort backend/' to fix them."
    exit 1
fi

# Run black to check formatting
echo "🖤 Checking code formatting with black..."
uv run black backend/ --check --diff

if [ $? -ne 0 ]; then
    echo "❌ Code formatting issues found. Run 'uv run black backend/' to fix them."
    exit 1
fi

# Run flake8 for linting
echo "🔍 Running linting with flake8..."
uv run flake8 backend/

if [ $? -ne 0 ]; then
    echo "❌ Linting issues found. Please fix the issues above."
    exit 1
fi

echo "✅ All code quality checks passed!"