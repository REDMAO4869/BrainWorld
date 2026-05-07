from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from generate_fake_data import build_pairs_from_manifests, generate_fake_voxels


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = REPO_ROOT / "examples" / "smoke" / "runtime"
TMP_ROOT = REPO_ROOT / ".tmp_smoke_run"


def _run(cmd, env):
    print("[smoke]", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    if TMP_ROOT.exists():
        shutil.rmtree(TMP_ROOT)
    if RUNTIME_ROOT.exists():
        shutil.rmtree(RUNTIME_ROOT)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)

    fake = generate_fake_voxels(RUNTIME_ROOT)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env["DATA_ROOT"] = str(fake["data_root"])
    env["OUTPUT_ROOT"] = str(RUNTIME_ROOT / "outputs")
    env["WEIGHTS_ROOT"] = str(RUNTIME_ROOT / "weights")
    env["CACHE_ROOT"] = str(RUNTIME_ROOT / "cache")
    env["MPLBACKEND"] = "Agg"

    runtime_cfg_root = RUNTIME_ROOT / "configs"
    vae_train_cfg = _load_json(REPO_ROOT / "configs" / "vae" / "train.template.json")
    vae_train_cfg["device"] = "cpu"
    vae_train_cfg["data"]["train"] = [str(fake["split_csvs"]["train"])]
    vae_train_cfg["data"]["val"] = [str(fake["split_csvs"]["val"])]
    vae_train_cfg["data"]["test"] = [str(fake["split_csvs"]["test"])]
    vae_train_cfg["data"]["t_frames"] = 4
    vae_train_cfg["model"].update(
        {
            "stage_channels": [8, 12, 16, 20],
            "latent_dim": 4,
            "res_blocks_per_stage": 1,
            "spatial_kernel_size": 3,
            "temporal_kernel_size": 3,
            "wavelet_energy_channels": 4,
            "wavelet_stages": ["enc2", "dec1"],
            "num_spatial_downsamples": 3,
            "temporal_mode": "spatial_only",
        }
    )
    vae_train_cfg["training"].update(
        {
            "num_workers": 0,
            "epochs": 1,
            "use_amp": False,
            "use_ddp": False,
            "gpu_ids": [],
            "output_root": str(RUNTIME_ROOT / "outputs" / "vae_train"),
        }
    )
    vae_train_path = runtime_cfg_root / "vae_train_smoke.json"
    _write_json(vae_train_path, vae_train_cfg)
    _run([sys.executable, "-m", "brainworld.vae.train", "--config", str(vae_train_path)], env)

    ckpt_path = next((RUNTIME_ROOT / "outputs" / "vae_train").glob("*/checkpoints/best.pt"), None)
    if ckpt_path is None:
        ckpt_path = next((RUNTIME_ROOT / "outputs" / "vae_train").glob("*/checkpoints/epoch_001.pt"))
    (RUNTIME_ROOT / "weights").mkdir(parents=True, exist_ok=True)
    vae_weight = RUNTIME_ROOT / "weights" / "vae_checkpoint.pt"
    shutil.copy2(ckpt_path, vae_weight)

    manifest_paths = {}
    for split in ("train", "val", "test"):
        ex_cfg = _load_json(REPO_ROOT / "configs" / "vae" / "extract.template.json")
        ex_cfg["device"] = "cpu"
        ex_cfg["data"]["train"] = [str(fake["split_csvs"]["train"])]
        ex_cfg["data"]["val"] = [str(fake["split_csvs"]["val"])]
        ex_cfg["data"]["test"] = [str(fake["split_csvs"]["test"])]
        ex_cfg["data"]["t_frames"] = 4
        ex_cfg["model"] = dict(vae_train_cfg["model"])
        ex_cfg["extraction"].update(
            {
                "checkpoint": str(vae_weight),
                "split": split,
                "batch_size": 1,
                "num_workers": 0,
                "output_dir": str(RUNTIME_ROOT / "outputs" / "vae_latents"),
                "gpu_ids": [],
                "use_amp": False,
                "use_timestamped_output": False,
            }
        )
        ex_path = runtime_cfg_root / f"vae_extract_{split}.json"
        _write_json(ex_path, ex_cfg)
        _run([sys.executable, "-m", "brainworld.vae.extract", "--config", str(ex_path)], env)
        manifest_paths[split] = RUNTIME_ROOT / "outputs" / "vae_latents" / "_state" / split / "manifest.csv"

    pairs_root = fake["data_root"] / "pairs"
    for split in ("train", "val", "test"):
        build_pairs_from_manifests(manifest_paths[split], pairs_root / f"{split}.csv", split)

    dit_train_cfg = _load_json(REPO_ROOT / "configs" / "dit" / "train.template.json")
    dit_train_cfg["device"] = "cpu"
    for split in ("train", "val", "test"):
        dit_train_cfg["data"]["target"][split] = [str(pairs_root / f"{split}.csv")]
    dit_train_cfg["model"].update({"patch_size": [1, 1, 1], "hidden_dim": 32, "depth": 2, "num_heads": 4, "dropout": 0.0})
    dit_train_cfg["diffusion"]["num_steps"] = 10
    dit_train_cfg["training"].update({"output_root": str(RUNTIME_ROOT / "outputs" / "dit_train"), "epochs": 1, "batch_size": 1, "num_workers": 0, "gpu_ids": []})
    dit_train_path = runtime_cfg_root / "dit_train_smoke.json"
    _write_json(dit_train_path, dit_train_cfg)
    _run([sys.executable, "-m", "brainworld.dit.train", "--config", str(dit_train_path)], env)

    dit_ckpt = next((RUNTIME_ROOT / "outputs" / "dit_train").glob("*/checkpoints/latest.pt"), None)
    if dit_ckpt is None:
        dit_ckpt = next((RUNTIME_ROOT / "outputs" / "dit_train").glob("*/checkpoints/best.pt"))
    dit_weight = RUNTIME_ROOT / "weights" / "dit_checkpoint.pt"
    shutil.copy2(dit_ckpt, dit_weight)

    dit_infer_cfg = _load_json(REPO_ROOT / "configs" / "dit" / "infer.template.json")
    dit_infer_cfg["device"] = "cpu"
    dit_infer_cfg["data"] = dit_train_cfg["data"]
    dit_infer_cfg["model"] = dit_train_cfg["model"]
    dit_infer_cfg["diffusion"]["num_steps"] = 10
    dit_infer_cfg["inference"].update({"checkpoint": str(dit_weight), "output_dir": str(RUNTIME_ROOT / "outputs" / "dit_infer"), "max_samples": 2, "batch_size": 1, "num_workers": 0})
    dit_infer_path = runtime_cfg_root / "dit_infer_smoke.json"
    _write_json(dit_infer_path, dit_infer_cfg)
    _run([sys.executable, "-m", "brainworld.dit.infer", "--config", str(dit_infer_path)], env)

    downstream_cfg = _load_json(REPO_ROOT / "configs" / "dit" / "downstream.template.json")
    downstream_cfg["device"] = "cpu"
    downstream_cfg["ckpt"]["checkpoint"] = str(dit_weight)
    downstream_cfg["data"] = dit_train_cfg["data"]
    downstream_cfg["output"]["out_root"] = str(RUNTIME_ROOT / "outputs" / "downstream")
    downstream_cfg["output"]["cache_root"] = str(RUNTIME_ROOT / "cache" / "downstream")
    downstream_cfg["linear_probe_cache"]["root"] = str(RUNTIME_ROOT / "cache" / "downstream")
    downstream_cfg["label"]["label_csv_path"] = str(fake["labels_path"])
    downstream_cfg["model"] = dit_train_cfg["model"]
    downstream_cfg["diffusion"] = dit_train_cfg["diffusion"]
    downstream_cfg["tasks"] = [{"name": "smoke_binary", "enabled": True, "mode": "linear_probe", "train": {"batch_size": 2, "best_metric": "f1_weighted"}}]
    downstream_path = runtime_cfg_root / "dit_downstream_smoke.json"
    _write_json(downstream_path, downstream_cfg)
    _run([sys.executable, "-m", "brainworld.dit.downstream", "--config", str(downstream_path)], env)

    shutil.rmtree(fake["data_root"], ignore_errors=True)
    shutil.rmtree(runtime_cfg_root, ignore_errors=True)
    success_path = RUNTIME_ROOT / "smoke_success.json"
    success_path.write_text(json.dumps({"status": "ok"}, indent=2), encoding="utf-8")
    print("[smoke] success", flush=True)


if __name__ == "__main__":
    main()
