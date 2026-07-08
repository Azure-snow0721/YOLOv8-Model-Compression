from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import ensure_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot accuracy-speed/complexity tradeoff curves.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def annotate_points(ax, df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        ax.annotate(
            f"{row['model']}\n{int(row['imgsz'])}",
            (row["fps"], row["map50"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    from common import abs_path

    metrics_path = Path(args.metrics) if args.metrics else abs_path(cfg, "project") / "metrics" / "metrics.csv"
    out_dir = ensure_dir(Path(args.output_dir) if args.output_dir else abs_path(cfg, "project") / "figures")
    df = pd.read_csv(metrics_path)

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(9, 6))
    for model_name, part in df.groupby("model"):
        part = part.sort_values("fps")
        ax.plot(part["fps"], part["map50"], marker="o", linewidth=2, label=model_name)
    annotate_points(ax, df)
    ax.set_xlabel("FPS")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("精度-速度权衡曲线")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_speed_tradeoff.png", dpi=300)

    fig, ax = plt.subplots(figsize=(9, 6))
    for model_name, part in df.groupby("model"):
        part = part.sort_values("flops_g")
        ax.plot(part["flops_g"], part["map50"], marker="o", linewidth=2, label=model_name)
    for _, row in df.iterrows():
        ax.annotate(f"{int(row['imgsz'])}", (row["flops_g"], row["map50"]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("FLOPs (G)")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("精度-计算量权衡曲线")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_flops_tradeoff.png", dpi=300)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric, ylabel in [
        (axes[0], "map50", "mAP@0.5"),
        (axes[1], "flops_g", "FLOPs (G)"),
        (axes[2], "fps", "FPS"),
    ]:
        pivot = df.pivot(index="model", columns="imgsz", values=metric)
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(ylabel)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.suptitle("416x416 与 640x640 输入分辨率影响")
    fig.tight_layout()
    fig.savefig(out_dir / "resolution_comparison.png", dpi=300)

    summary_path = out_dir / "metrics_table.md"
    table = df.sort_values(["model", "imgsz"]).to_markdown(index=False, floatfmt=".4f")
    summary_path.write_text(table + "\n", encoding="utf-8")
    print(f"Saved figures and table to: {out_dir}")


if __name__ == "__main__":
    main()
