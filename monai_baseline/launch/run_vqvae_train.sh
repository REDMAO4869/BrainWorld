#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
CONFIG=${CONFIG:-${PROJECT_ROOT}/configs/vqvae_stage1_train.json}
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
    ('manifest_paths', 'train'),
    ('manifest_paths', 'val'),
    ('output_dir',),
    ('warm_start', 'checkpoint_path'),
])
summary = {
    'config': str(config_path),
    'output_dir': cfg['output_dir'],
    'train_manifest': cfg['manifest_paths']['train'],
    'val_manifest': cfg['manifest_paths']['val'],
    'train_manifest_exists': Path(cfg['manifest_paths']['train']).exists(),
    'val_manifest_exists': Path(cfg['manifest_paths']['val']).exists(),
    'warm_start_checkpoint': cfg.get('warm_start', {}).get('checkpoint_path'),
    'warm_start_checkpoint_exists': Path(cfg['warm_start']['checkpoint_path']).exists() if cfg.get('warm_start', {}).get('checkpoint_path') else False,
    'use_ddp': bool(cfg.get('training', {}).get('use_ddp', False)),
    'model': {
        'spatial_dims': cfg.get('model', {}).get('spatial_dims'),
        'channels': cfg.get('model', {}).get('channels'),
        'num_channels': cfg.get('model', {}).get('num_channels'),
        'num_res_layers': cfg.get('model', {}).get('num_res_layers'),
        'num_embeddings': cfg.get('model', {}).get('num_embeddings'),
        'embedding_dim': cfg.get('model', {}).get('embedding_dim'),
    },
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
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}"                 "${PROJECT_ROOT}/scripts/train_vqvae_stage1.py" --config "${CONFIG}" "$@"
else
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train_vqvae_stage1.py" --config "${CONFIG}" "$@"
fi
