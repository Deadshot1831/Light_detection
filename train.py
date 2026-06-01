import os
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

# Reduce CUDA memory fragmentation (recommended by PyTorch for OOM errors)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "Vehicle light detection.v2-head_tail_lights-v1.yolov11"
SRC_YAML = DATASET_DIR / "data.yaml"
RESOLVED_YAML = DATASET_DIR / "data.resolved.yaml"


def build_resolved_yaml() -> Path:
    with SRC_YAML.open("r") as f:
        cfg = yaml.safe_load(f)

    cfg["path"] = str(DATASET_DIR)
    cfg["train"] = "train/images"
    cfg["val"] = "valid/images"
    cfg["test"] = "test/images"

    with RESOLVED_YAML.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return RESOLVED_YAML


def main() -> None:
    data_yaml = build_resolved_yaml()

    # Free any leftover GPU memory from previous runs
    torch.cuda.empty_cache()

    model = YOLO("yolo26m.pt")

    model.train(
        data=str(data_yaml),
        epochs=40,
        imgsz=640,
        batch=8,  # reduced from 16 to avoid CUDA OOM (increase once GPU memory is free)
        patience=20,
        optimizer="auto",
        lr0=0.01,
        cos_lr=True,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name="yolo26m_vehicle_lights",
        exist_ok=True,
        pretrained=True,
        seed=42,
        device=0,
        workers=8,
        amp=True,
        plots=True,
    )

    model.val(data=str(data_yaml), split="test", imgsz=640)


if __name__ == "__main__":
    main()
