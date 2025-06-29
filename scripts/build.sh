#!/usr/bin/env bash
# This script sets up a Python virtual environment and installs the necessary dependencies.

if [ ! -d .venv ]; then
    # Create a virtual environment in the .venv directory if it doesn't exist
    python3 -m venv .venv
    # Activate the virtual environment
    source .venv/bin/activate
    # Upgrade pip to the latest version
    pip install --upgrade pip
    # Install the package in editable mode with development dependencies
    pip install -r requirements.txt
    # Print a message indicating that the virtual environment has been created
    echo ".venv created"
    # List the installed packages in the virtual environment
    pip list
fi


