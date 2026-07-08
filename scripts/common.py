from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(cfg_path)
    cfg["_root"] = str(ROOT)
    return cfg


def abs_path(cfg: dict[str, Any], key: str) -> Path:
    path = Path(cfg[key])
    if path.is_absolute():
        return path
    return Path(cfg["_root"]) / path


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: Any) -> str:
    value = str(device).strip().lower()
    if value in {"auto", ""}:
        return "0" if torch.cuda.is_available() else "cpu"
    if value != "cpu" and not torch.cuda.is_available():
        print(f"CUDA device={device} requested, but torch.cuda.is_available() is False. Falling back to CPU.")
        return "cpu"
    return str(device)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def best_pt(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    candidate = run_dir / "weights" / "best.pt"
    if candidate.exists():
        return candidate
    last = run_dir / "weights" / "last.pt"
    if last.exists():
        return last
    raise FileNotFoundError(f"No best.pt/last.pt found under {run_dir}")


def latest_run(project: str | Path, prefix: str) -> Path:
    project = Path(project)
    runs = sorted(project.glob(f"{prefix}*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"No run directory matched {project / (prefix + '*')}")
    return runs[0]


def latest_checkpoint(prefix: str, filename: str = "last.pt") -> Path:
    matches = sorted(
        ROOT.glob(f"runs/**/{prefix}*/weights/{filename}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No checkpoint matched runs/**/{prefix}*/weights/{filename}")
    return matches[0]
