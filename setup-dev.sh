#!/bin/bash
# Development environment setup script for Logsqueak
# This script creates a virtual environment and installs all dependencies
#
# Usage:
#   ./setup-dev.sh         # Interactive mode
#   ./setup-dev.sh --yes   # Skip all prompts (always yes)

set -e  # Exit on error

# Check for --yes flag
AUTO_YES=false
if [ "$1" == "--yes" ] || [ "$1" == "-y" ]; then
    AUTO_YES=true
fi

echo "=== Logsqueak Development Environment Setup ==="
echo ""

# Find Python 3.11 or later
PYTHON_CMD=""
for cmd in python3.13 python3.12 python3.11 python3; do
    if command -v $cmd &> /dev/null; then
        VERSION=$($cmd --version 2>&1 | awk '{print $2}')
        MAJOR=$(echo $VERSION | cut -d. -f1)
        MINOR=$(echo $VERSION | cut -d. -f2)

        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 11 ]; then
            PYTHON_CMD=$cmd
            echo "✓ Found Python $VERSION ($cmd)"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Python 3.11 or later is required"
    echo "Please install Python 3.11+ and try again"
    exit 1
fi

echo ""

# Create virtual environment if it doesn't exist
if [ -d "venv" ]; then
    echo "⚠ Virtual environment already exists at ./venv"
    if [ "$AUTO_YES" = false ]; then
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Removing existing venv..."
            rm -rf venv
        else
            echo "Keeping existing venv"
        fi
    else
        echo "Keeping existing venv (--yes mode)"
    fi
fi

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    echo "✓ Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Verify we're in the venv
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Error: Failed to activate virtual environment"
    exit 1
fi

echo "✓ Virtual environment activated: $VIRTUAL_ENV"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"
echo ""

# Install packages in editable mode with all dependencies
echo "Installing logsqueak in editable mode with dev dependencies..."
pip install -e ".[dev]" --quiet
echo "✓ logsqueak and dev dependencies installed"
echo ""

echo "Reinstalling logseq-outline-parser from local source..."
pip install -e src/logseq-outline-parser/ --force-reinstall --no-deps --quiet
echo "✓ logseq-outline-parser updated from source"
echo ""

echo "=== Setup Complete! ==="
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest -v                # All tests"
echo "  pytest -n auto           # All tests in parallel"
echo "  pytest tests/unit/ -v    # Unit tests only"
echo ""
echo "To deactivate the virtual environment:"
echo "  deactivate"
echo ""
