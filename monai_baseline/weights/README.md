# Weights

This public bundle does not ship real checkpoints.

Place the real files at these relative destinations before running extraction or inference:

- `weights/checkpoints/vqvae_best.pt`
- `weights/checkpoints/dit_stage2_nextonly_best.pt`

The metadata for the selected checkpoints is recorded in `weights_manifest.json`.

Expected behavior:

- `scripts/build_latent_cache.py` expects `vqvae_best.pt` to contain `model_state_dict`.
- `scripts/infer_dit_stage2_next.py` expects `dit_stage2_nextonly_best.pt` to contain `model_state_dict`, and optionally `ema_model_state_dict` when EMA sampling is enabled.
