# MONAI Baseline

MONAI Baseline is a lightweight public baseline for 4D fMRI tokenization and next-window latent prediction built around MONAI-style training scripts. It provides a compact reference implementation for stage-1 VQ-VAE training, latent extraction, stage-2 diffusion training, and inference utilities.

This repository is intended to accompany the main `BrainWorld` codebase as a simpler baseline with a smaller surface area and script-first workflow.

## Components

- `scripts/train_vqvae_stage1.py`: stage-1 VQ-VAE training
- `scripts/build_latent_cache.py`: 4D latent extraction and caching
- `scripts/train_dit_stage2.py`: stage-2 next-window latent prediction training
- `scripts/infer_dit_stage2_next.py`: stage-2 inference
- `scripts/build_exp1_universal_split.py`: split preparation for paired latent and FC-condition inputs
- `launch/*.sh`: launcher scripts for training, extraction, split building, and inference

## Repository Layout

- `src/monai_fmri_public/`: shared helpers for config loading, data handling, models, EMA, and visualization
- `scripts/`: runnable Python entrypoints
- `launch/`: shell wrappers for common workflows
- `configs/`: public JSON templates for stage-1, extraction, and stage-2 jobs
- `examples/`: example manifests, metadata, and split-layout templates
- `weights/`: placeholder checkpoint locations plus a manifest for recommended files
- `docs/`: anonymization and release notes for the baseline package

## Quickstart

```bash
bash launch/run_vqvae_train.sh --dry-run
bash launch/run_vqvae_extract.sh --dry-run
bash launch/run_prepare_split.sh --dry-run
bash launch/run_dit_train.sh --dry-run
bash launch/run_dit_infer.sh --dry-run
```

The launchers accept environment-variable overrides such as `CONFIG`, `PYTHON_BIN`, `OUT_ROOT`, `LATENT_ROOT`, and `FC_ROOT`.

## Weights

Real checkpoints are not tracked in the repository. Expected checkpoint destinations are documented in `weights/README.md`, and placeholder files are kept only to preserve the intended relative layout.

## Data and Privacy

This baseline does not ship private datasets, internal machine paths, user identifiers, host information, or environment-specific secrets. Public example manifests use placeholder dataset names and relative paths only.
