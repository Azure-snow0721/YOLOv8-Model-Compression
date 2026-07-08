from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch_pruning as tp
from ultralytics import YOLO

from common import abs_path, best_pt, ensure_dir, latest_run, load_config, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structured channel pruning for YOLOv8n with torch-pruning.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--weights", default=None, help="Path to baseline YOLOv8n best.pt")
    parser.add_argument("--ratio", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--finetune-epochs", type=int, default=None)
    return parser.parse_args()


def collect_ignored_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    ignored = []
    for module in model.modules():
        if module.__class__.__name__ in {"Detect", "Segment", "Pose", "OBB"}:
            ignored.append(module)
    return ignored


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    yolo_device = resolve_device(cfg["device"])
    device = torch.device("cuda:0" if yolo_device != "cpu" else "cpu")
    imgsz = args.imgsz or int(cfg["imgsz_train"])
    ratio = args.ratio if args.ratio is not None else float(cfg["prune_ratio"])
    data_yaml = abs_path(cfg, "yolo_dataset_dir") / "voc2012.yaml"
    weights = Path(args.weights) if args.weights else best_pt(latest_run(cfg["project"], "baseline_yolov8n"))

    yolo = YOLO(str(weights))
    net = yolo.model.to(device)
    net.train()
    example_inputs = torch.randn(1, 3, imgsz, imgsz, device=device)
    ignored_layers = collect_ignored_layers(net)

    base_macs, base_params = tp.utils.count_ops_and_params(net, example_inputs)
    importance = tp.importance.MagnitudeImportance(p=2)
    pruner = tp.pruner.MagnitudePruner(
        net,
        example_inputs,
        importance=importance,
        iterative_steps=1,
        pruning_ratio=ratio,
        ignored_layers=ignored_layers,
        round_to=8,
    )
    pruner.step()
    pruned_macs, pruned_params = tp.utils.count_ops_and_params(net, example_inputs)

    out_dir = ensure_dir(abs_path(cfg, "project") / "pruned_yolov8n_tp" / "weights")
    raw_path = out_dir / "pruned_raw.pt"
    yolo.save(str(raw_path))
    print(f"Saved pruned raw model: {raw_path}")
    print(f"Params: {base_params / 1e6:.2f}M -> {pruned_params / 1e6:.2f}M")
    print(f"FLOPs: {base_macs * 2 / 1e9:.2f}G -> {pruned_macs * 2 / 1e9:.2f}G")

    finetune_epochs = args.finetune_epochs
    if finetune_epochs is None:
        finetune_epochs = int(cfg["epochs_prune_finetune"])
    if finetune_epochs > 0:
        yolo.train(
            data=str(data_yaml),
            imgsz=imgsz,
            epochs=finetune_epochs,
            batch=int(cfg["batch"]),
            workers=int(cfg["workers"]),
            device=yolo_device,
            project=str(abs_path(cfg, "project")),
            name="pruned_yolov8n_tp_finetune",
            exist_ok=True,
            seed=int(cfg["seed"]),
        )


if __name__ == "__main__":
    main()
