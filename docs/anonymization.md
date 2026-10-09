# Anonymization Notes

This public package removes private machine paths, usernames, hostnames, and internal project roots.

Public configs should use environment variables instead of absolute private paths:

- `DATA_ROOT`
- `OUTPUT_ROOT`
- `WEIGHTS_ROOT`
- `CACHE_ROOT`

The JSON loader expands `${VAR_NAME}` and `__REPO_ROOT__` automatically.

## Runtime Artifacts

Smoke tests and local experiments may create temporary manifests, fake samples, checkpoints, caches, and summaries under local runtime folders. These artifacts are validation byproducts and should not be committed to the public repository.

The default ignore rules cover common generated paths such as:

- `examples/smoke/runtime/`
- `.tmp_smoke_run/`
- `weights/assets/`
- `__pycache__/`
