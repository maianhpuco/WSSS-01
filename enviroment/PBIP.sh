#!/usr/bin/env bash
# Minimal environment setup for PBIP

set -euo pipefail

# --- Config ---
ENV_NAME="pbip"
PYTHON_VERSION="3.9"

# --- Resolve requirements.txt ---
if [[ -f "./src/externals/PBIP/requirements.txt" ]]; then
  REQ_FILE="$(cd "./src/externals/PBIP" && pwd)/requirements.txt"
elif [[ -f "requirements.txt" ]]; then
  REQ_FILE="$(pwd)/requirements.txt"
else
  echo "requirements.txt not found"
  echo "Create one or update REQ_FILE path in this script."
  exit 1
fi

# --- Conda availability ---
if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found. Please install Miniconda/Anaconda first."
  exit 1
fi

# Enable 'conda activate' in non-interactive shells
eval "$(conda shell.bash hook)"

# --- Create env if needed ---
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Creating conda env '$ENV_NAME' (Python $PYTHON_VERSION)..."
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
else
  echo "Conda env '$ENV_NAME' already exists. Skipping creation."
fi

# --- Activate & install ---
echo "Activating env '$ENV_NAME'..."
conda activate "$ENV_NAME"

echo "Upgrading pip/setuptools/wheel..."
python -m pip install -U pip setuptools wheel

echo "Installing requirements from: $REQ_FILE"
pip install -r "$REQ_FILE"

# other dependencies
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
pip install icecream

echo
echo "Environment ready and activated: $ENV_NAME"
echo "   python: $(command -v python)"
echo "   pip:    $(command -v pip)"
