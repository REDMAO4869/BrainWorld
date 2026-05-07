# BrainWorld

BrainWorld is a research codebase for 4D fMRI representation learning and conditional generation. The overall pipeline has two stages:

1. `VAE stage`: compress 4D fMRI volumes from voxel space into a latent space, with support for latent extraction and decoding.
2. `DiT stage`: perform conditional diffusion modeling in latent space for training and generation.

The repository includes the core code, public config templates, launch scripts, and supporting documentation for training, inference, and further research development.

A MONAI-style baseline is maintained separately in a companion repository, for example `../monai_baseline`.

## Repository Layout

- `vae/`
  - VAE training, latent extraction, and decoding code
- `dit/`
  - Conditional DiT training and inference code
- `configs/`
  - Public config templates
- `scripts/`
  - Common launch scripts
- `manifests/`
  - Data manifest templates and examples
- `weights/`
  - Checkpoint notes and placeholder paths
- `docs/`
  - Supporting documentation
- `outputs/`
  - Runtime output directory

## Main Workflow

The recommended workflow is:

1. Train the VAE
2. Extract latent representations
3. Train the conditional DiT
4. Run inference or generation
5. Decode latents back to voxel space when needed

In short:

- `VAE` handles representation compression and reconstruction
- `DiT` handles conditional generation in latent space

## Quick Start

You can run the main Python entrypoints directly:

```bash
python vae/train_wf_vae.py --config configs/vae/wf_vae_train_spatial_only.json
python vae/extract_wf_latents.py --config configs/vae/wf_vae_extract_spatial_only_train.json
python dit/train_cond_dit.py --config configs/dit/cond_dit_train_spatial_only_public.json
python dit/infer_cond_dit.py --config configs/dit/cond_dit_infer_public.json
```

You can also use the launch scripts under `scripts/`.

## Weights

Recommended checkpoint paths:

- `weights/vae/best.pt`
- `weights/dit/best.pt`

For checkpoint download and distribution details, see `docs/weight_distribution.md`.

## Notes

- `configs/` provides public config templates
- `manifests/` provides data manifest templates and examples
- `outputs/` stores runtime results
