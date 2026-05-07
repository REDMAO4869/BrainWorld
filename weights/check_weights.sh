#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$SCRIPT_DIR/manifest.json"
ASSET_DIR="${1:-$SCRIPT_DIR/assets}"
python3 - "$MANIFEST" "$ASSET_DIR" <<'PY'
import hashlib, json, os, sys
manifest_path, asset_dir = sys.argv[1:3]
with open(manifest_path, 'r', encoding='utf-8') as f:
    entries = json.load(f)
failed = False
for entry in entries:
    path = os.path.join(asset_dir, entry['release_asset'])
    if not os.path.isfile(path):
        print(f"[check] missing {path}")
        failed = True
        continue
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    got = h.hexdigest()
    ok = got == entry['sha256']
    print(f"[check] {entry['release_asset']} {'OK' if ok else 'FAIL'}")
    if not ok:
        print(f"  expected={entry['sha256']}")
        print(f"  got     ={got}")
        failed = True
if failed:
    sys.exit(1)
PY
