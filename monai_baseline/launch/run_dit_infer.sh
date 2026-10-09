#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
CONFIG=${CONFIG:-${PROJECT_ROOT}/configs/dit_stage2_infer_nextonly.json}
DRY_RUN=${DRY_RUN:-0}
CHECKPOINT=${CHECKPOINT:-}

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  CONFIG_PATH="${CONFIG}" PROJECT_ROOT_ENV="${PROJECT_ROOT}" CHECKPOINT_ENV="${CHECKPOINT}" "${PYTHON_BIN}" - <<'PY'
import json
import sys
from pathlib import Path

project_root = Path(__import__('os').environ['PROJECT_ROOT_ENV'])
config_path = Path(__import__('os').environ['CONFIG_PATH']).resolve()
checkpoint_override = __import__('os').environ.get('CHECKPOINT_ENV', '').strip()
sys.path.insert(0, str(project_root / 'src'))
from monai_fmri_public.config import load_json, resolve_paths

cfg = load_json(config_path)
cfg = resolve_paths(cfg, config_path.parent, [
    ('task_vocab_path',),
    ('latent_stats_path',),
    ('output_dir',),
    ('data', 'universal_split_root'),
    ('infer', 'checkpoint_path'),
    ('infer', 'output_root'),
])
ckpt = checkpoint_override or str(cfg.get('infer', {}).get('checkpoint_path') or '')
task_vocab_path = Path(cfg['task_vocab_path'])
summary = {
    'config': str(config_path),
    'checkpoint': ckpt,
    'checkpoint_exists': Path(ckpt).exists() if ckpt else False,
    'task_vocab_path': cfg['task_vocab_path'],
    'task_vocab_exists': task_vocab_path.exists(),
    'latent_stats_path': cfg['latent_stats_path'],
    'latent_stats_exists': Path(cfg['latent_stats_path']).exists(),
    'split_root': cfg['data']['universal_split_root'],
    'split_root_exists': Path(cfg['data']['universal_split_root']).exists(),
    'infer_output_root': cfg['infer']['output_root'],
    'model_type': cfg.get('model_type', 'dit3d'),
    'prediction_type': cfg.get('diffusion', {}).get('prediction_type'),
    'pair_mode': cfg.get('conditioning', {}).get('pair_mode'),
}
print(json.dumps(summary, indent=2))
PY
  exit 0
fi

if [[ -z "${CHECKPOINT}" ]]; then
  CHECKPOINT=$(CONFIG_PATH="${CONFIG}" PROJECT_ROOT_ENV="${PROJECT_ROOT}" "${PYTHON_BIN}" - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str((Path(__import__('os').environ['PROJECT_ROOT_ENV']) / 'src').resolve()))
from monai_fmri_public.config import load_json, resolve_paths
config_path = Path(__import__('os').environ['CONFIG_PATH']).resolve()
cfg = resolve_paths(load_json(config_path), config_path.parent, [('infer', 'checkpoint_path')])
print(cfg.get('infer', {}).get('checkpoint_path', '') or '')
PY
  )
fi

if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "[error] checkpoint not found: ${CHECKPOINT:-<empty>}" >&2
  exit 1
fi

USE_DDP=$(CONFIG_PATH="${CONFIG}" "${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
cfg = json.loads(Path(__import__('os').environ['CONFIG_PATH']).read_text(encoding='utf-8'))
print('1' if bool(cfg.get('training', {}).get('use_ddp', False)) else '0')
PY
)

ARGS=(--config "${CONFIG}" --checkpoint "${CHECKPOINT}")
if [[ "${USE_DDP}" == "1" && "${NPROC_PER_NODE:-1}" -gt 1 ]]; then
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}"                 "${PROJECT_ROOT}/scripts/infer_dit_stage2_next.py" "${ARGS[@]}" "$@"
else
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/infer_dit_stage2_next.py" "${ARGS[@]}" "$@"
fi
