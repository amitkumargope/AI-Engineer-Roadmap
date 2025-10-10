#!/usr/bin/env python
"""
ppe_realtime.py — Real-time PPE detection with YOLOv8

Usage examples:
  python ppe_realtime.py --weights runs/ppe_yolov8/weights/best.pt --source 0
  python ppe_realtime.py --weights runs/ppe_yolov8/weights/best.pt --source "C:\path\to\file.mp4"
  python ppe_realtime.py --weights runs/ppe_yolov8/weights/best.pt --source "rtsp://user:pass@ip:554/stream"

Press 'q' to quit the window.
"""
import argparse
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

def run(weights, source, conf=0.25, imgsz=640):
    model = YOLO(weights)
    cap = cv2.VideoCapture(source if source != "0" else 0)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)
            res = results[0]
            annotator = Annotator(frame, line_width=2)
            if res.boxes is not None and len(res.boxes) > 0:
                for box in res.boxes:
                    b = box.xyxy[0].cpu().numpy().astype(int)
                    cls_id = int(box.cls[0].item())
                    confv = float(box.conf[0].item())
                    label = f"{model.names.get(cls_id, cls_id)} {confv:.2f}"
                    annotator.box_label(b, label, color=colors(cls_id, True))

            cv2.imshow("YOLOv8 PPE — Realtime (press q to quit)", annotator.result())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="Path to trained weights .pt (e.g., runs/ppe_yolov8/weights/best.pt)")
    ap.add_argument("--source", type=str, default="0", help="0 for webcam, or path/URL to video/RTSP")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    run(args.weights, args.source, conf=args.conf, imgsz=args.imgsz)
