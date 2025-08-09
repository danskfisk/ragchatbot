#!/usr/bin/env bash

# Auto-fix code formatting script

echo "🔧 Auto-fixing code formatting..."

# Sort imports with isort
echo "📦 Sorting imports with isort..."
uv run isort backend/

# Format code with black
echo "🖤 Formatting code with black..."
uv run black backend/

echo "✅ Code formatting complete!"
echo "💡 Run './scripts/format.sh' to verify all quality checks pass."