#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
CONFIG=${CONFIG:-${PROJECT_ROOT}/configs/vqvae_extract_4d.json}
DRY_RUN=${DRY_RUN:-0}

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  CONFIG_PATH="${CONFIG}" PROJECT_ROOT_ENV="${PROJECT_ROOT}" "${PYTHON_BIN}" - <<'PY'
import json
import sys
from pathlib import Path

project_root = Path(__import__('os').environ['PROJECT_ROOT_ENV'])
config_path = Path(__import__('os').environ['CONFIG_PATH']).resolve()
sys.path.insert(0, str(project_root / 'src'))
from monai_fmri_public.config import load_json, resolve_paths

cfg = load_json(config_path)
cfg = resolve_paths(cfg, config_path.parent, [
    ('vqvae_checkpoint',),
    ('manifest_paths', 'train'),
    ('manifest_paths', 'val'),
    ('manifest_paths', 'test'),
    ('cache', 'output_root'),
])
summary = {
    'config': str(config_path),
    'vqvae_checkpoint': cfg['vqvae_checkpoint'],
    'vqvae_checkpoint_exists': Path(cfg['vqvae_checkpoint']).exists(),
    'cache_output_root': cfg['cache']['output_root'],
    'manifest_paths': cfg['manifest_paths'],
}
print(json.dumps(summary, indent=2))
PY
  exit 0
fi

USE_DDP=$(CONFIG_PATH="${CONFIG}" "${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
cfg = json.loads(Path(__import__('os').environ['CONFIG_PATH']).read_text(encoding='utf-8'))
print('1' if bool(cfg.get('training', {}).get('use_ddp', False)) else '0')
PY
)

if [[ "${USE_DDP}" == "1" && "${NPROC_PER_NODE:-1}" -gt 1 ]]; then
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}"                 "${PROJECT_ROOT}/scripts/build_latent_cache.py" --config "${CONFIG}" "$@"
else
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_latent_cache.py" --config "${CONFIG}" "$@"
fi
