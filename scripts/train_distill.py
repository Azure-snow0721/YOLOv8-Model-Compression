from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml
from tqdm import tqdm
from ultralytics import YOLO

from common import abs_path, best_pt, latest_run, load_config, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher-to-student YOLOv8 pseudo-label distillation.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--teacher", default=None, help="Path to trained YOLOv8m teacher .pt")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=None)
    return parser.parse_args()


def yolo_iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax1, ay1, ax2, ay2 = ax - aw / 2, ay - ah / 2, ax + aw / 2, ay + ah / 2
    bx1, by1, bx2, by2 = bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def read_labels(path: Path) -> list[tuple[int, list[float]]]:
    if not path.exists():
        return []
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            labels.append((int(float(parts[0])), [float(v) for v in parts[1:5]]))
    return labels


def write_labels(path: Path, labels: list[tuple[int, list[float]]]) -> None:
    text = "\n".join(f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}" for cls, box in labels)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def build_distill_dataset(cfg: dict, teacher_path: Path, imgsz: int, conf: float, device: str) -> Path:
    src = abs_path(cfg, "yolo_dataset_dir")
    dst = src.parent / "voc2012_yolo_distill"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    train_images = sorted((dst / "images" / "train").glob("*.jpg"))
    label_dir = dst / "labels" / "train"
    teacher = YOLO(str(teacher_path))
    duplicate_iou = float(cfg["distill_iou_duplicate"])

    for image_path in tqdm(train_images, desc="teacher pseudo labels"):
        label_path = label_dir / f"{image_path.stem}.txt"
        merged = read_labels(label_path)
        result = teacher.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=conf,
            iou=0.6,
            verbose=False,
            device=device,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue
        xywhn = result.boxes.xywhn.cpu().tolist()
        cls_ids = result.boxes.cls.cpu().int().tolist()
        for cls_id, box in zip(cls_ids, xywhn):
            duplicate = any(cls_id == old_cls and yolo_iou(box, old_box) >= duplicate_iou for old_cls, old_box in merged)
            if not duplicate:
                merged.append((cls_id, box))
        write_labels(label_path, merged)

    data = yaml.safe_load((src / "voc2012.yaml").read_text(encoding="utf-8"))
    data["path"] = str(dst.resolve()).replace("\\", "/")
    with (dst / "voc2012_distill.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return dst


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg["device"])
    teacher_path = Path(args.teacher) if args.teacher else best_pt(latest_run(cfg["project"], "teacher_yolov8m"))
    imgsz = args.imgsz or int(cfg["imgsz_train"])
    conf = args.conf if args.conf is not None else float(cfg["distill_conf"])
    distill_dir = build_distill_dataset(cfg, teacher_path, imgsz, conf, device)

    student = YOLO(cfg["baseline_model"])
    student.train(
        data=str(distill_dir / "voc2012_distill.yaml"),
        imgsz=imgsz,
        epochs=args.epochs or int(cfg["epochs_distill"]),
        batch=int(cfg["batch"]),
        workers=int(cfg["workers"]),
        device=device,
        project=str(abs_path(cfg, "project")),
        name="distill_yolov8n",
        exist_ok=True,
        seed=int(cfg["seed"]),
        pretrained=True,
    )


if __name__ == "__main__":
    main()
