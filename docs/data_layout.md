# Public Data Layout

## VAE inputs

VAE training expects 4D fMRI arrays in `.npy` or `.npz` files plus split CSVs whose `path` column points to files or directories.

## DIT inputs

DIT training expects direct-pair CSVs with at least:

- `Subject`
- `sequence_id`
- `anchor_chunk_id`
- `target_chunk_id`
- `pair_direction`
- `vae_latent_path`
- `target_latent_path`
- `fc_embedding_path`
- `MRI_embedding_path`

The smoke workflow under `examples/smoke/` shows the minimum runnable format.
