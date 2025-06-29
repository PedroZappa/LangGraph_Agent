#!/usr/bin/env bash
# This script runs the application

source .venv/bin/activate

if [ ! -f .env ]; then
  touch .env
fi

source .env


