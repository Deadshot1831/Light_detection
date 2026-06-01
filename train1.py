from pathlib import Path

import yaml
from ultralytics.models import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "Vehicle light detection.v3-head_tail_lights-v2--960-px-.yolov11"
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

    model = YOLO("yolo11m.pt")

    model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=960,
        batch=16,
        patience=10,
        optimizer="auto",
        lr0=0.01,
        cos_lr=True,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name="yolo11x_vehicle_lights",
        exist_ok=True,
        pretrained=True,
        device=0,
        workers=4,
        plots=True,
    )

    model.val(data=str(data_yaml), split="test", imgsz=960)


if __name__ == "__main__":
    main()
