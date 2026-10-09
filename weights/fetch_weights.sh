#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$SCRIPT_DIR/manifest.json"
OUT_DIR="${1:-$SCRIPT_DIR/assets}"
RELEASE_BASE_URL="${BRAINWORLD_RELEASE_BASE_URL:-}"
if [[ -z "$RELEASE_BASE_URL" ]]; then
  echo "Set BRAINWORLD_RELEASE_BASE_URL to the GitHub Release asset base URL." >&2
  exit 1
fi
mkdir -p "$OUT_DIR"
python3 - "$MANIFEST" "$OUT_DIR" "$RELEASE_BASE_URL" <<'PY'
import json, os, sys, urllib.request
manifest_path, out_dir, base = sys.argv[1:4]
with open(manifest_path, 'r', encoding='utf-8') as f:
    entries = json.load(f)
for entry in entries:
    asset = entry['release_asset']
    url = base.rstrip('/') + '/' + asset
    dst = os.path.join(out_dir, asset)
    print(f'[fetch] {url} -> {dst}', flush=True)
    urllib.request.urlretrieve(url, dst)
PY
