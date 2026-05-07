from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a BrainWorld generation config from a training snapshot")
    p.add_argument("--train-snapshot", required=True, help="Path to a saved training config snapshot JSON")
    p.add_argument("--checkpoint", required=True, help="Path to a trained DIT checkpoint")
    p.add_argument("--output-dir", required=True, help="Directory for generated samples")
    p.add_argument("--out-config", required=True, help="Where to write the generated JSON config")
    p.add_argument("--num-steps", type=int, default=1000, help="Diffusion steps for generation")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=1)
    p.add_argument("--cfg-scale", type=float, default=2.0)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.train_snapshot)
    out = Path(args.out_config)
    cfg = json.loads(src.read_text(encoding="utf-8"))
    cfg["device"] = args.device
    cfg.setdefault("diffusion", {})["num_steps"] = int(args.num_steps)
    cfg["inference"] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "auto_load_model_config": False,
        "split": "test",
        "batch_size": int(args.batch_size),
        "num_workers": 0,
        "max_samples": int(args.max_samples),
        "save_dtype": "float32",
        "clip_x0": None,
        "output_dir": str(Path(args.output_dir).resolve()),
        "cfg": {"enabled": True, "scale": float(args.cfg_scale), "base_mode": "fc_only"},
        "decoder": {"enabled": False},
        "condition_ablation": {"enabled": False},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
