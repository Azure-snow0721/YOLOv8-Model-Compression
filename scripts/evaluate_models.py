from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch_pruning as tp
from ultralytics import YOLO

from common import abs_path, best_pt, ensure_dir, latest_run, load_config, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate mAP@0.5, FLOPs, params and FPS at multiple resolutions.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--models", nargs="*", default=None, help="Optional name=path entries.")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def resolve_default_models(cfg: dict[str, Any]) -> dict[str, Path]:
    project = abs_path(cfg, "project")
    candidates: dict[str, Path] = {}
    for name, prefix in [
        ("YOLOv8n-baseline", "baseline_yolov8n"),
        ("YOLOv8n-distill", "distill_yolov8n"),
        ("YOLOv8n-pruned", "pruned_yolov8n_tp_finetune"),
        ("YOLOv8m-teacher", "teacher_yolov8m"),
    ]:
        try:
            candidates[name] = best_pt(latest_run(project, prefix))
        except FileNotFoundError:
            pass
    raw_pruned = project / "pruned_yolov8n_tp" / "weights" / "pruned_raw.pt"
    if "YOLOv8n-pruned" not in candidates and raw_pruned.exists():
        candidates["YOLOv8n-pruned"] = raw_pruned
    return candidates


def parse_model_entries(entries: list[str]) -> dict[str, Path]:
    models = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Model entry must be name=path, got: {entry}")
        name, path = entry.split("=", 1)
        models[name] = Path(path)
    return models


def complexity(model: YOLO, imgsz: int, device: str) -> tuple[float, float]:
    net = model.model
    net.eval()
    torch_device = torch.device("cuda:0" if device != "cpu" and torch.cuda.is_available() else "cpu")
    net.to(torch_device)
    dummy = torch.randn(1, 3, imgsz, imgsz, device=torch_device)
    macs, params = tp.utils.count_ops_and_params(net, dummy)
    flops_g = macs * 2 / 1e9
    params_m = params / 1e6
    return flops_g, params_m


def benchmark_fps(model: YOLO, image_paths: list[Path], imgsz: int, cfg: dict[str, Any], device: str) -> float:
    warmup = int(cfg["fps_warmup"])
    iters = int(cfg["fps_iters"])
    if not image_paths:
        return 0.0
    samples = [str(p) for p in image_paths[: max(1, min(len(image_paths), iters))]]
    for i in range(warmup):
        model.predict(samples[i % len(samples)], imgsz=imgsz, device=device, verbose=False)
    if torch.cuda.is_available() and str(device) != "cpu":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(iters):
        model.predict(samples[i % len(samples)], imgsz=imgsz, device=device, verbose=False)
    if torch.cuda.is_available() and str(device) != "cpu":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return iters / elapsed if elapsed > 0 else 0.0


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    data_yaml = abs_path(cfg, "yolo_dataset_dir") / "voc2012.yaml"
    eval_split = str(cfg.get("eval_split", "val"))
    test_images = sorted((abs_path(cfg, "yolo_dataset_dir") / "images" / eval_split).glob("*.jpg"))
    output = Path(args.output) if args.output else abs_path(cfg, "project") / "metrics" / "metrics.csv"
    ensure_dir(output.parent)

    models = parse_model_entries(args.models) if args.models else resolve_default_models(cfg)
    if not models:
        raise FileNotFoundError("No trained models found. Train baseline/distill/pruned models first.")

    rows = []
    for model_name, weight_path in models.items():
        for imgsz in cfg["imgsz_eval"]:
            print(f"Evaluating {model_name} @ {imgsz}: {weight_path}")
            model = YOLO(str(weight_path))
            metrics = model.val(
                data=str(data_yaml),
                split=eval_split,
                imgsz=int(imgsz),
                batch=int(cfg["batch"]),
                device=device,
                project=str(abs_path(cfg, "project")),
                name=f"val_{model_name}_{imgsz}".replace("/", "_"),
                exist_ok=True,
                verbose=False,
            )
            flops_g, params_m = complexity(model, int(imgsz), device)
            fps = benchmark_fps(model, test_images, int(imgsz), cfg, device)
            rows.append(
                {
                    "model": model_name,
                    "weights": str(weight_path),
                    "imgsz": int(imgsz),
                    "eval_split": eval_split,
                    "map50": float(metrics.box.map50),
                    "flops_g": flops_g,
                    "params_m": params_m,
                    "fps": fps,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(df)
    print(f"Saved metrics: {output}")


if __name__ == "__main__":
    main()
