#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
OUT_ROOT=${OUT_ROOT:-${PROJECT_ROOT}/artifacts/universal_split}
LATENT_ROOT=${LATENT_ROOT:-${PROJECT_ROOT}/artifacts/vqvae_latents}
FC_ROOT=${FC_ROOT:-${PROJECT_ROOT}/data/fc_embeddings}
DRY_RUN=${DRY_RUN:-0}

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  OUT_ROOT_ENV="${OUT_ROOT}" LATENT_ROOT_ENV="${LATENT_ROOT}" FC_ROOT_ENV="${FC_ROOT}" "${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
import os

summary = {
    'output_root': str(Path(os.environ['OUT_ROOT_ENV']).resolve()),
    'latent_root': str(Path(os.environ['LATENT_ROOT_ENV']).resolve()),
    'fc_root': str(Path(os.environ['FC_ROOT_ENV']).resolve()),
}
print(json.dumps(summary, indent=2))
PY
  exit 0
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_exp1_universal_split.py"               --output-root "${OUT_ROOT}"               --latent-root "${LATENT_ROOT}"               --fc-root "${FC_ROOT}"               "$@"
