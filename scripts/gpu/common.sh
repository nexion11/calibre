# Shared setup for every GPU job. Edit the three lines marked EDIT once, for your cluster account.
# On USC CARC (Discovery) the account name is your PI's project allocation, e.g. "zada_123".
set -euo pipefail
CALIBRE_DIR="${CALIBRE_DIR:-$HOME/calibre}"   # EDIT if the repo lives elsewhere (CARC: /project/<account>/calibre is roomier)
cd "$CALIBRE_DIR"
export PATH="$HOME/.local/bin:$PATH"           # where the uv installer puts uv
export CALIBRE_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
