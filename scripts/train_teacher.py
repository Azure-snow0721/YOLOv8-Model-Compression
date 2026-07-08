from __future__ import annotations

import argparse

from ultralytics import YOLO

from common import abs_path, latest_checkpoint, load_config, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8m teacher on VOC2012.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    data_yaml = abs_path(cfg, "yolo_dataset_dir") / "voc2012.yaml"
    device = resolve_device(cfg["device"])
    model = YOLO(cfg["teacher_model"])
    if args.resume:
        model = YOLO(str(latest_checkpoint("teacher_yolov8m", "last.pt")))
        model.train(resume=True)
    else:
        model.train(
            data=str(data_yaml),
            imgsz=args.imgsz or int(cfg["imgsz_train"]),
            epochs=args.epochs or int(cfg["epochs_teacher"]),
            batch=int(cfg["batch"]),
            workers=int(cfg["workers"]),
            device=device,
            project=str(abs_path(cfg, "project")),
            name="teacher_yolov8m",
            exist_ok=True,
            seed=int(cfg["seed"]),
            pretrained=True,
            cache=False,
        )


if __name__ == "__main__":
    main()
