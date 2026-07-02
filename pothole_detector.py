"""Pothole detection utilities using Ultralytics YOLO.

Dependencies:
  pip install ultralytics torch torchvision opencv-python pillow matplotlib

Optional dependencies:
  pip install albumentations roboflow fastapi uvicorn python-multipart

Usage examples:
  python pothole_detector.py train --model yolov8s.pt --data data.yaml --epochs 100 --batch 8
  python pothole_detector.py val --model runs/detect/pothole_detector/weights/best.pt --data data.yaml
  python pothole_detector.py predict --model runs/detect/pothole_detector/weights/best.pt --source test_images/ --conf 0.5 --save
  python pothole_detector.py webcam --model runs/detect/pothole_detector/weights/best.pt --conf 0.5
  python pothole_detector.py export --model runs/detect/pothole_detector/weights/best.pt --format onnx
"""

import argparse
from pathlib import Path
from typing import List, Optional

from ultralytics import YOLO


class PotholeDetector:
    def __init__(self, model_path: str = "yolov8s.pt", device: str = "0"):
        self.model_path = model_path
        self.device = device
        self.model = YOLO(model_path)

    @staticmethod
    def create_data_yaml(
        dataset_root: str,
        train_dir: str = "train/images",
        val_dir: str = "valid/images",
        test_dir: Optional[str] = None,
        nc: int = 1,
        names: Optional[List[str]] = None,
    ) -> str:
        root = Path(dataset_root).expanduser().resolve()
        names = names or ["pothole"]
        data_yaml = root / "data.yaml"
        paths = {
            "train": train_dir,
            "val": val_dir,
            "test": test_dir or "test/images",
        }

        with data_yaml.open("w", encoding="utf-8") as f:
            f.write(f"path: {root.as_posix()}\n")
            f.write(f"train: {paths['train']}\n")
            f.write(f"val: {paths['val']}\n")
            if test_dir is not None:
                f.write(f"test: {paths['test']}\n")
            f.write(f"nc: {nc}\n")
            f.write("names:\n")
            for idx, name in enumerate(names):
                f.write(f"  {idx}: {name}\n")

        return str(data_yaml)

    def train(
        self,
        data: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 8,
        lr0: float = 0.001,
        lrf: float = 0.01,
        optimizer: str = "AdamW",
        name: str = "pothole_detector",
        patience: int = 20,
        save_period: int = 10,
        device: Optional[str] = None,
    ) -> None:
        run_device = device or self.device
        print(f"Training model={self.model_path} data={data} device={run_device}")
        self.model.train(
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            optimizer=optimizer,
            lr0=lr0,
            lrf=lrf,
            name=name,
            patience=patience,
            save_period=save_period,
            device=run_device,
        )
        print("Training finished.")

    def validate(self, data: str) -> None:
        print(f"Validating model={self.model_path} data={data}")
        metrics = self.model.val(data=data)
        try:
            box = metrics.box
            print(f"mAP@50: {box.map50:.4f}")
            print(f"mAP@50:95: {box.map:.4f}")
            print(f"Precision: {box.mp:.4f}")
            print(f"Recall: {box.mr:.4f}")
        except Exception:
            print(metrics)

    def predict(
        self,
        source: str,
        conf: float = 0.5,
        save: bool = False,
        save_dir: str = "runs/detect/predict",
        show: bool = False,
    ) -> None:
        print(f"Predicting source={source} model={self.model_path}")
        results = self.model(source, conf=conf, save=save, show=show, save_dir=save_dir)
        if isinstance(results, list):
            for r in results:
                print(f"{Path(r.path).name}: {len(r.boxes)} detections")
        else:
            print(f"{Path(results.path).name}: {len(results.boxes)} detections")

    def predict_webcam(self, conf: float = 0.5, show: bool = True) -> None:
        print(f"Running webcam inference with model={self.model_path}")
        self.model(source=0, conf=conf, show=show)

    def export(self, export_format: str = "onnx", simplify: bool = False) -> None:
        print(f"Exporting model={self.model_path} format={export_format}")
        self.model.export(format=export_format, simplify=simplify)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pothole detection with YOLO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a YOLO model")
    train_parser.add_argument("--model", default="yolov8s.pt", help="Base YOLO weights or config")
    train_parser.add_argument("--data", required=True, help="Path to data.yaml")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--imgsz", type=int, default=640)
    train_parser.add_argument("--batch", type=int, default=8)
    train_parser.add_argument("--lr0", type=float, default=0.001)
    train_parser.add_argument("--lrf", type=float, default=0.01)
    train_parser.add_argument("--optimizer", default="AdamW")
    train_parser.add_argument("--name", default="pothole_detector")
    train_parser.add_argument("--patience", type=int, default=20)
    train_parser.add_argument("--save-period", type=int, default=10)
    train_parser.add_argument("--device", default="0")

    val_parser = subparsers.add_parser("val", help="Validate a trained model")
    val_parser.add_argument("--model", required=True, help="Path to trained weights")
    val_parser.add_argument("--data", required=True, help="Path to data.yaml")

    predict_parser = subparsers.add_parser("predict", help="Run inference on images or video")
    predict_parser.add_argument("--model", required=True, help="Path to trained weights")
    predict_parser.add_argument("--source", required=True, help="Image, directory, or video source")
    predict_parser.add_argument("--conf", type=float, default=0.5)
    predict_parser.add_argument("--save", action="store_true")
    predict_parser.add_argument("--show", action="store_true")
    predict_parser.add_argument("--save-dir", default="runs/detect/predict")

    webcam_parser = subparsers.add_parser("webcam", help="Run real-time webcam detection")
    webcam_parser.add_argument("--model", required=True, help="Path to trained weights")
    webcam_parser.add_argument("--conf", type=float, default=0.5)

    export_parser = subparsers.add_parser("export", help="Export a trained model")
    export_parser.add_argument("--model", required=True, help="Path to trained weights")
    export_parser.add_argument("--format", default="onnx", choices=["onnx", "engine", "tflite", "torch", "onnx", "coreml", "pmml"])
    export_parser.add_argument("--simplify", action="store_true")

    yaml_parser = subparsers.add_parser("create_yaml", help="Create a data.yaml for YOLO training")
    yaml_parser.add_argument("--dataset-root", required=True, help="Root dataset directory")
    yaml_parser.add_argument("--train-dir", default="train/images")
    yaml_parser.add_argument("--val-dir", default="valid/images")
    yaml_parser.add_argument("--test-dir", default="test/images")
    yaml_parser.add_argument("--names", default="pothole", help="Comma-separated class names")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "create_yaml":
        names = [name.strip() for name in args.names.split(",") if name.strip()]
        yaml_path = PotholeDetector.create_data_yaml(
            dataset_root=args.dataset_root,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            test_dir=args.test_dir,
            names=names,
        )
        print(f"Created data.yaml at: {yaml_path}")
        return

    detector = PotholeDetector(model_path=args.model)

    if args.command == "train":
        detector.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            lr0=args.lr0,
            lrf=args.lrf,
            optimizer=args.optimizer,
            name=args.name,
            patience=args.patience,
            save_period=args.save_period,
            device=args.device,
        )
    elif args.command == "val":
        detector.validate(data=args.data)
    elif args.command == "predict":
        detector.predict(
            source=args.source,
            conf=args.conf,
            save=args.save,
            save_dir=args.save_dir,
            show=args.show,
        )
    elif args.command == "webcam":
        detector.predict_webcam(conf=args.conf)
    elif args.command == "export":
        detector.export(export_format=args.format, simplify=args.simplify)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
