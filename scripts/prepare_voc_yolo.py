from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from tqdm import tqdm

from common import abs_path, ensure_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Pascal VOC2012 detection data to YOLO format.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def voc_box_to_yolo(size: tuple[int, int], box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    width, height = size
    xmin, ymin, xmax, ymax = box
    x = ((xmin + xmax) / 2.0) / width
    y = ((ymin + ymax) / 2.0) / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    return x, y, w, h


def convert_split(voc_dir: Path, split_file: Path, out_dir: Path, split: str, class_to_id: dict[str, int]) -> int:
    image_out = ensure_dir(out_dir / "images" / split)
    label_out = ensure_dir(out_dir / "labels" / split)
    ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    written = 0
    for image_id in tqdm(ids, desc=f"convert {split}"):
        xml_path = voc_dir / "Annotations" / f"{image_id}.xml"
        jpg_path = voc_dir / "JPEGImages" / f"{image_id}.jpg"
        if not xml_path.exists() or not jpg_path.exists():
            continue
        root = ET.parse(xml_path).getroot()
        size_node = root.find("size")
        width = int(size_node.findtext("width"))
        height = int(size_node.findtext("height"))
        labels: list[str] = []
        for obj in root.findall("object"):
            cls_name = obj.findtext("name")
            difficult = int(obj.findtext("difficult", default="0"))
            if cls_name not in class_to_id or difficult == 1:
                continue
            bndbox = obj.find("bndbox")
            xmin = max(0.0, float(bndbox.findtext("xmin")))
            ymin = max(0.0, float(bndbox.findtext("ymin")))
            xmax = min(float(width), float(bndbox.findtext("xmax")))
            ymax = min(float(height), float(bndbox.findtext("ymax")))
            if xmax <= xmin or ymax <= ymin:
                continue
            x, y, w, h = voc_box_to_yolo((width, height), (xmin, ymin, xmax, ymax))
            labels.append(f"{class_to_id[cls_name]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        if not labels:
            continue
        shutil.copy2(jpg_path, image_out / jpg_path.name)
        (label_out / f"{image_id}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
        written += 1
    return written


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = abs_path(cfg, "yolo_dataset_dir")
    data_yaml_path = out_dir / "voc2012.yaml"
    required_dirs = [
        out_dir / "images" / "train",
        out_dir / "images" / "val",
        out_dir / "images" / "test",
        out_dir / "labels" / "train",
        out_dir / "labels" / "val",
        out_dir / "labels" / "test",
    ]
    if data_yaml_path.exists() and all(p.exists() for p in required_dirs) and not args.overwrite:
        print(f"Existing YOLO dataset found, skip conversion: {out_dir}")
        print("Use --overwrite if you want to rebuild it.")
        return
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)

    train_voc = abs_path(cfg, "train_voc_dir")
    test_voc = abs_path(cfg, "test_voc_dir")
    classes = cfg["classes"]
    class_to_id = {name: idx for idx, name in enumerate(classes)}

    train_count = convert_split(
        train_voc,
        train_voc / "ImageSets" / "Main" / "train.txt",
        out_dir,
        "train",
        class_to_id,
    )
    val_count = convert_split(
        train_voc,
        train_voc / "ImageSets" / "Main" / "val.txt",
        out_dir,
        "val",
        class_to_id,
    )
    test_count = convert_split(
        test_voc,
        test_voc / "ImageSets" / "Main" / "test.txt",
        out_dir,
        "test",
        class_to_id,
    )

    data_yaml = {
        "path": str(out_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(classes),
        "names": classes,
    }
    with (out_dir / "voc2012.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"YOLO dataset written to: {out_dir}")
    print(f"train={train_count}, val={val_count}, test={test_count}")


if __name__ == "__main__":
    main()
