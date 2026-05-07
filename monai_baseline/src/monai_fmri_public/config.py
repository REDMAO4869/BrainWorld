from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def save_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_resolved_config(config: dict[str, Any], output_dir: str | Path) -> None:
    save_json(Path(output_dir) / "resolved_config.json", config)


def resolve_path(value: str | Path | None, base_dir: str | Path) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "null":
        return value
    path = Path(text)
    if not path.is_absolute():
        path = (Path(base_dir) / path).resolve()
    return str(path)


def _get_nested(payload: dict[str, Any], key_path: Sequence[str]) -> Any:
    current: Any = payload
    for key in key_path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _set_nested(payload: dict[str, Any], key_path: Sequence[str], value: Any) -> None:
    current = payload
    for key in key_path[:-1]:
        current = current.setdefault(key, {})
    current[key_path[-1]] = value


def resolve_paths(config: dict[str, Any], base_dir: str | Path, key_paths: Iterable[Sequence[str]]) -> dict[str, Any]:
    for key_path in key_paths:
        current = _get_nested(config, key_path)
        if current is None:
            continue
        if isinstance(current, str) and current.strip() == "":
            continue
        resolved = resolve_path(current, base_dir)
        _set_nested(config, key_path, resolved)
    return config
