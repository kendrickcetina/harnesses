#!/usr/bin/env bash
# Environment for the Comillas DGX. Run at the start of EVERY session, from the repo root:
#
#     source scripts/dgx_env.sh
#
# It keeps uv, Python, the Hugging Face cache and the CLI tools inside
# /home/<user>/clusters/dgx, the only folder that survives between sessions.

WORK_DIR="${HARNESS_WORK_DIR:-$HOME/clusters/dgx}"

if [ ! -d "$WORK_DIR" ]; then
    echo "Cannot find $WORK_DIR. Are you on the DGX? Otherwise export HARNESS_WORK_DIR first."
    return 1 2>/dev/null || exit 1
fi

export UV_INSTALL_DIR="$WORK_DIR/.uv/bin"
export UV_CACHE_DIR="$WORK_DIR/.uv/cache"
export UV_PYTHON_INSTALL_DIR="$WORK_DIR/.uv/python"
export HF_HOME="$WORK_DIR/.hf_cache"
export HARNESS_TOOLS_DIR="$WORK_DIR/.harness_tools"
export GIT_CONFIG_GLOBAL="$WORK_DIR/.gitconfig"
export TOKENIZERS_PARALLELISM=false
touch "$GIT_CONFIG_GLOBAL"
mkdir -p "$UV_INSTALL_DIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$HF_HOME" "$HARNESS_TOOLS_DIR/bin"
export PATH="$HARNESS_TOOLS_DIR/bin:$UV_INSTALL_DIR:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv in $UV_INSTALL_DIR ..."
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_INSTALL_DIR" UV_NO_MODIFY_PATH=1 sh
fi

[ -f .env ] && { set -a; . ./.env; set +a; }

echo "uv:      $(uv --version 2>/dev/null || echo 'not available')"
echo "HF_HOME: $HF_HOME"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none visible')"
echo "codex:   $(command -v codex || echo 'not installed (optional: bash scripts/install_clis.sh)')"
echo "claude:  $(command -v claude || echo 'not installed (optional: bash scripts/install_clis.sh)')"
