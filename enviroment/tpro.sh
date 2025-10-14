#!/usr/bin/env bash
# Minimal environment setup for TPRO

set -euo pipefail

# --- Config ---
ENV_NAME="tpro"
PYTHON_VERSION="3.9"

# --- Paths ---
TPRO_DIR="./src/externals/TPRO"
TPRO_REQ_FILE="$TPRO_DIR/requirements.txt"


conda install -y pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch -c conda-forge 


if [[ ! -d "$TPRO_DIR" ]]; then
  echo "TPRO directory not found at $TPRO_DIR"
  echo "Make sure the submodule is initialized: git submodule update --init --recursive"
  exit 1
fi

if [[ ! -f "$TPRO_REQ_FILE" ]]; then
  echo "requirements.txt not found at $TPRO_REQ_FILE"
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

echo "Preparing filtered requirements (exclude JAX-related, not needed by TPRO)..."
TMP_REQ="$(mktemp)"
grep -v -E '^(jax(==|>=|<=)?|jaxlib(==|>=|<=)?|flax(==|>=|<=)?|optax(==|>=|<=)?|chex(==|>=|<=)?|dm-tree(==|>=|<=)?|tensorstore(==|>=|<=)?)' "$TPRO_REQ_FILE" > "$TMP_REQ"

echo "Installing core requirements from: $TPRO_REQ_FILE (filtered)"
pip install -r "$TMP_REQ"

# Additional direct-from-git deps commonly required in this repo
pip install git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1
pip install git+https://github.com/huggingface/transformers@539e2281cd97c35ef4122757f26c88f44115fa94

# Optional: GPU-specific torch versions (uncomment and adjust to your CUDA)
# pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

echo
echo "TPRO environment ready and activated: $ENV_NAME"
echo "   python: $(command -v python)"
echo "   pip:    $(command -v pip)"
echo "   note: JAX-related packages were intentionally skipped."


