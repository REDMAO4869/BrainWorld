# Weight Packaging

Large checkpoints are expected to be distributed through GitHub Release assets.

- `weights/manifest.json` stores metadata and SHA256 digests.
- `weights/fetch_weights.sh` downloads named assets from a release base URL.
- `weights/check_weights.sh` verifies local files against the manifest.
