#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
CONFIG=${CONFIG:-${PROJECT_ROOT}/configs/dit_stage2_train_nextonly.json}
DRY_RUN=${DRY_RUN:-0}
PREPARE_SPLIT=${PREPARE_SPLIT:-0}
AUTO_RESUME=${AUTO_RESUME:-1}

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

if [[ "${PREPARE_SPLIT}" == "1" ]]; then
  OUT_ROOT=$(CONFIG_PATH="${CONFIG}" PROJECT_ROOT_ENV="${PROJECT_ROOT}" "${PYTHON_BIN}" - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str((Path(__import__('os').environ['PROJECT_ROOT_ENV']) / 'src').resolve()))
from monai_fmri_public.config import load_json, resolve_paths
config_path = Path(__import__('os').environ['CONFIG_PATH']).resolve()
cfg = resolve_paths(load_json(config_path), config_path.parent, [('data', 'universal_split_root')])
print(cfg['data']['universal_split_root'])
PY
  )
  OUT_ROOT="${OUT_ROOT}" "${PROJECT_ROOT}/launch/run_prepare_split.sh"
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
    ('task_vocab_path',),
    ('latent_stats_path',),
    ('output_dir',),
    ('data', 'universal_split_root'),
    ('stage2_visualization', 'vqvae_model_config_json'),
    ('stage2_visualization', 'vqvae_checkpoint'),
])

task_vocab_path = Path(cfg['task_vocab_path'])
summary = {
    'config': str(config_path),
    'output_dir': cfg['output_dir'],
    'task_vocab_path': cfg['task_vocab_path'],
    'task_vocab_exists': task_vocab_path.exists(),
    'latent_stats_path': cfg['latent_stats_path'],
    'latent_stats_exists': Path(cfg['latent_stats_path']).exists(),
    'split_root': cfg['data']['universal_split_root'],
    'split_root_exists': Path(cfg['data']['universal_split_root']).exists(),
    'datasets': cfg['data'].get('datasets', []),
    'model_type': cfg.get('model_type', 'dit3d'),
    'prediction_type': cfg.get('diffusion', {}).get('prediction_type'),
    'pair_mode': cfg.get('conditioning', {}).get('pair_mode'),
    'conditioning': {
        'use_anchor_fc_condition': bool(cfg.get('conditioning', {}).get('use_anchor_fc_condition', False)),
        'fc_dim_hint': cfg.get('conditioning', {}).get('fc', {}).get('dim_hint'),
    },
}
print(json.dumps(summary, indent=2))
PY
  exit 0
fi

RESUME_FROM=$(CONFIG_PATH="${CONFIG}" PROJECT_ROOT_ENV="${PROJECT_ROOT}" "${PYTHON_BIN}" - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str((Path(__import__('os').environ['PROJECT_ROOT_ENV']) / 'src').resolve()))
from monai_fmri_public.config import load_json, resolve_paths
config_path = Path(__import__('os').environ['CONFIG_PATH']).resolve()
cfg = resolve_paths(load_json(config_path), config_path.parent, [('resume_from',), ('output_dir',)])
resume = cfg.get('resume_from')
output_dir = Path(cfg['output_dir'])
if resume:
    print(resume)
elif output_dir.joinpath('diffusion_last.pt').exists():
    print(str(output_dir / 'diffusion_last.pt'))
else:
    print('')
PY
)

USE_DDP=$(CONFIG_PATH="${CONFIG}" "${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
cfg = json.loads(Path(__import__('os').environ['CONFIG_PATH']).read_text(encoding='utf-8'))
print('1' if bool(cfg.get('training', {}).get('use_ddp', False)) else '0')
PY
)

ARGS=(--config "${CONFIG}")
if [[ -n "${RESUME_FROM}" && "${AUTO_RESUME}" == "1" ]]; then
  ARGS+=(--resume-from "${RESUME_FROM}")
fi

if [[ "${USE_DDP}" == "1" && "${NPROC_PER_NODE:-1}" -gt 1 ]]; then
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}"                 "${PROJECT_ROOT}/scripts/train_dit_stage2.py" "${ARGS[@]}" "$@"
else
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train_dit_stage2.py" "${ARGS[@]}" "$@"
fi
