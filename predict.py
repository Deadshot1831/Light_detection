import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "yolo26m_vehicle_lights"
    / "weights"
    / "best.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOv26 inference on an image.")
    parser.add_argument("source", help="Path to image (or directory/video).")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    model = YOLO(args.weights)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project=str(PROJECT_ROOT / "runs" / "predict"),
        name="yolo26m_vehicle_lights",
        exist_ok=True,
    )

    for r in results:
        print(f"\nImage: {r.path}")
        print(f"Saved to: {r.save_dir}")
        if r.boxes is None or len(r.boxes) == 0:
            print("  No detections.")
            continue
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = [round(v, 1) for v in box.xyxy[0].tolist()]
            print(f"  {names[cls_id]:<11} conf={conf:.2f}  bbox={xyxy}")


if __name__ == "__main__":
    main()
