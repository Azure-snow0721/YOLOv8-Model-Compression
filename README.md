# YOLOv8 Model Compression on VOC2012

This project implements a complete YOLOv8 object detection compression pipeline based on the Pascal VOC2012 dataset. It includes:

- Baseline model training
- Teacher model training
- Structured model pruning
- Knowledge distillation
- Model evaluation
- Result visualization

The scripts are implemented using the Ultralytics YOLO framework and PyTorch.

---

## Project Structure

```
project/
│
├── scripts/
│   ├── common.py
│   ├── prepare_voc_yolo.py
│   ├── train_baseline.py
│   ├── train_teacher.py
│   ├── prune_model.py
│   ├── train_distill.py
│   ├── evaluate_models.py
│   └── plot_results.py
│
├── runs/
│   └── detect/
│
├── config.yaml
├── requirements.txt
├── README.md
└── CONTRIBUTION.txt
```

---

## Features

- YOLOv8 baseline training
- Teacher model training
- Model pruning
- Knowledge distillation
- Automatic evaluation
- Result plotting
- Reproducible experiments

---

## Installation

Create a Python environment:

```bash
python -m venv venv
```

Activate it.

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

This project uses the Pascal VOC2012 dataset.

Prepare the dataset first:

```bash
python scripts/prepare_voc_yolo.py
```

The dataset location can be configured in `config.yaml`.

---

## Training

### Train Baseline

```bash
python scripts/train_baseline.py
```

### Train Teacher

```bash
python scripts/train_teacher.py
```

### Prune Model

```bash
python scripts/prune_model.py
```

### Knowledge Distillation

```bash
python scripts/train_distill.py
```

---

## Evaluation

Evaluate trained models:

```bash
python scripts/evaluate_models.py
```

Generate result figures:

```bash
python scripts/plot_results.py
```

---

## Results

Training outputs are stored under:

```
runs/detect/
```

Including:

- checkpoints
- training curves
- confusion matrices
- precision-recall curves
- validation metrics

---

## Requirements

- Python 3.10+
- PyTorch
- Ultralytics YOLOv8

See `requirements.txt` for the full dependency list.

---

## License

This project is intended for academic research and educational purposes.
