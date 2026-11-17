#!/bin/bash
# One-time setup on a login node (no GPU needed). Run from the repo folder after copying it and data/ over.
#   bash scripts/gpu/setup.sh
set -euo pipefail
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync
mkdir -p logs
test -d data/frames/JPEGImages || { echo "data/frames/JPEGImages is missing: copy it from the Mac first"; exit 1; }
echo "frames: $(ls data/frames/JPEGImages | wc -l) (expect 31443)"
uv run python -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda)"
