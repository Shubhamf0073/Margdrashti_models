"""
Real-time YOLOv8 inference for pothole and crack detection with ROI support.

This script provides a drop-in replacement for deploy_inference_realtime_fast.py
using YOLOv8 object detection instead of tile-based classification.

Usage:
    # Basic usage
    python deploy_inference_yolov8.py \
        --model scripts/runs/yolov8n_stage2/weights/best.pt \
        --roi roi_highway_shorter.json \
        --source source/videos/demo.mp4

    # Save output video
    python deploy_inference_yolov8.py \
        --model scripts/runs/yolov8n_stage2/weights/best.pt \
        --roi roi_highway_shorter.json \
        --source source/videos/demo.mp4 \
        --save_output results/yolov8_output.mp4 \
        --confidence 0.5

    # Process from webcam
    python deploy_inference_yolov8.py \
        --model scripts/runs/yolov8n_stage2/weights/best.pt \
        --roi roi_highway_shorter.json \
        --source 0

    # Frame skipping for slower hardware
    python deploy_inference_yolov8.py \
        --model scripts/runs/yolov8n_stage2/weights/best.pt \
        --roi roi_highway_shorter.json \
        --source source/videos/demo.mp4 \
        --process_every 2
"""

import argparse
import json
import time
from pathlib import Path
from collections import deque
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Install with: pip install ultralytics")
    exit(1)


class YOLOv8RoadAuditInference:
    """
    Real-time YOLOv8 inference for pothole and crack detection.

    This class provides the same API as RoadAuditInference from
    deploy_inference_realtime_fast.py for seamless integration.
    """

    def __init__(self, model_path, roi_path, confidence_threshold=0.5, device='cuda'):
        """
        Initialize YOLOv8 inference pipeline.

        Args:
            model_path: Path to trained YOLOv8 model (.pt file)
            roi_path: Path to ROI JSON configuration
            confidence_threshold: Minimum confidence for detections (0-1)
            device: Device for inference ('cuda', 'cpu', 'mps', or 'auto')
        """
        self.device = self._pick_device(device)
        self.confidence_threshold = confidence_threshold

        # Load YOLOv8 model
        print(f"Loading YOLOv8 model from {model_path}...")
        self.model = YOLO(model_path)
        self.class_names = self.model.names  # Dict: {0: 'pothole', 1: 'crack', ...}

        print(f"Model loaded successfully")
        print(f"Classes: {list(self.class_names.values())}")
        print(f"Device: {self.device}")

        # Load ROI configuration
        with open(roi_path, 'r') as f:
            roi_data = json.load(f)
        self.roi_points = np.array(roi_data['roi_points'], dtype=np.int32)
        self.roi_mask = None

        # FPS tracking
        self.fps_buffer = deque(maxlen=30)

        # Color mapping for visualization
        self.colors = {
            'pothole': (0, 0, 255),           # Red
            'crack': (255, 165, 0),           # Orange
            'longitudinal_crack': (255, 255, 0),  # Yellow
            'transverse_crack': (0, 255, 255),    # Cyan
            'roi': (0, 255, 0)                # Green
        }

        # Default color for unknown classes
        self.default_color = (255, 255, 255)  # White

    def _pick_device(self, requested: str) -> str:
        """Determine the best available device."""
        import torch

        req = (requested or "auto").lower()
        if req == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if req == "mps":
            return "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        if req == "cpu":
            return "cpu"
        # Auto-detect
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def create_roi_mask(self, frame_shape):
        """Create binary mask from ROI polygon."""
        H, W = frame_shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [self.roi_points.reshape((-1, 1, 2))], 255)
        return mask

    def filter_detections_by_roi(self, detections):
        """
        Filter detections to only those inside ROI.

        Args:
            detections: List of detection dicts with 'bbox' key

        Returns:
            Filtered list of detections
        """
        if self.roi_mask is None:
            return detections

        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            # Check if bounding box center is inside ROI
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Bounds check
            if 0 <= cy < self.roi_mask.shape[0] and 0 <= cx < self.roi_mask.shape[1]:
                if self.roi_mask[cy, cx] > 0:
                    filtered.append(det)

        return filtered

    def process_frame(self, frame):
        """
        Process single frame with YOLOv8 detection.

        Args:
            frame: Input frame (BGR format)

        Returns:
            tuple: (annotated_frame, detections_list)
        """
        start_time = time.time()

        # Create ROI mask if needed
        if self.roi_mask is None:
            self.roi_mask = self.create_roi_mask(frame.shape)

        # Run YOLOv8 inference
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
            stream=False
        )

        # Parse detections
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    # Extract confidence and class
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.class_names.get(cls, f'class_{cls}')

                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2),
                        'timestamp': time.time()
                    })

        # Filter by ROI
        detections = self.filter_detections_by_roi(detections)

        # Annotate frame
        annotated = self.annotate_frame(frame, detections)

        # Update FPS
        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        self.fps_buffer.append(fps)

        return annotated, detections

    def annotate_frame(self, frame, detections):
        """
        Draw bounding boxes and labels on frame.

        Args:
            frame: Input frame (BGR format)
            detections: List of detection dicts

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        # Draw ROI polygon
        cv2.polylines(
            annotated,
            [self.roi_points.reshape((-1, 1, 2))],
            True,
            self.colors['roi'],
            2
        )

        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            confidence = det['confidence']

            # Get color for this class
            color = self.colors.get(class_name, self.default_color)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Prepare label
            label = f"{class_name}: {confidence:.2f}"

            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(y1, label_size[1] + 10)

            cv2.rectangle(
                annotated,
                (x1, label_y - label_size[1] - 10),
                (x1 + label_size[0], label_y),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # Draw FPS counter
        avg_fps = np.mean(self.fps_buffer) if len(self.fps_buffer) > 0 else 0
        cv2.putText(
            annotated,
            f"FPS: {avg_fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        # Draw detection count
        cv2.putText(
            annotated,
            f"Detections: {len(detections)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        # Draw per-class counts
        class_counts = {}
        for det in detections:
            cls = det['class']
            class_counts[cls] = class_counts.get(cls, 0) + 1

        y_offset = 120
        for cls, count in sorted(class_counts.items()):
            color = self.colors.get(cls, self.default_color)
            cv2.putText(
                annotated,
                f"{cls}: {count}",
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
            y_offset += 30

        return annotated

    def run(self, source, save_output=None, display=True, process_every=1):
        """
        Run inference on video source.

        Args:
            source: Video file path or camera index (0 for webcam)
            save_output: Path to save annotated video (optional)
            display: Show video window
            process_every: Process every Nth frame (for frame skipping)
        """
        # Open video source
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            cap = cv2.VideoCapture(int(source))
            print(f"Opened camera {source}")
        else:
            cap = cv2.VideoCapture(source)
            print(f"Opened video file: {source}")

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video: {width}×{height} @ {fps:.1f} FPS")

        # Setup video writer
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_output, fourcc, fps, (width, height))
            print(f"Saving output to: {save_output}")

        # Processing loop
        frame_count = 0
        total_detections = 0
        last_detections = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame or skip
                if process_every <= 1 or (frame_count % process_every == 0):
                    annotated, detections = self.process_frame(frame)
                    last_detections = detections
                else:
                    # Reuse last frame's detections
                    annotated = self.annotate_frame(frame, last_detections)
                    detections = last_detections

                total_detections += len(detections)

                # Log detections
                if len(detections) > 0:
                    det_summary = ', '.join([f"{d['class']} ({d['confidence']:.2f})" for d in detections])
                    print(f"Frame {frame_count}: {det_summary}")

                # Write to output
                if writer:
                    writer.write(annotated)

                # Display
                if display:
                    cv2.imshow('YOLOv8 Pothole/Crack Detection', annotated)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):
                        print("Quit requested")
                        break
                    elif key == ord('s'):
                        screenshot_path = f"screenshot_{frame_count:06d}.jpg"
                        cv2.imwrite(screenshot_path, annotated)
                        print(f"Saved screenshot: {screenshot_path}")
                    elif key == ord('p'):
                        print("Paused. Press any key to continue...")
                        cv2.waitKey(0)

                frame_count += 1

        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

            # Print summary
            avg_fps = np.mean(self.fps_buffer) if len(self.fps_buffer) > 0 else 0
            print("\n" + "="*60)
            print("INFERENCE COMPLETE")
            print("="*60)
            print(f"Frames processed:    {frame_count}")
            print(f"Total detections:    {total_detections}")
            print(f"Average FPS:         {avg_fps:.1f}")
            print(f"Detection rate:      {total_detections/frame_count if frame_count > 0 else 0:.2f} per frame")
            print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Real-time YOLOv8 inference for pothole and crack detection"
    )
    parser.add_argument(
        '--model',
        required=True,
        help='Path to trained YOLOv8 model (.pt file)'
    )
    parser.add_argument(
        '--roi',
        required=True,
        help='Path to ROI JSON configuration'
    )
    parser.add_argument(
        '--source',
        required=True,
        help='Video file path or camera index (0 for webcam)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Confidence threshold for detections (default: 0.5)'
    )
    parser.add_argument(
        '--save_output',
        type=str,
        help='Path to save annotated video'
    )
    parser.add_argument(
        '--no_display',
        action='store_true',
        help='Disable video display window'
    )
    parser.add_argument(
        '--process_every',
        type=int,
        default=1,
        help='Process every Nth frame (default: 1, no skipping)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help='Device for inference (default: auto)'
    )

    args = parser.parse_args()

    # Initialize inference
    inference = YOLOv8RoadAuditInference(
        model_path=args.model,
        roi_path=args.roi,
        confidence_threshold=args.confidence,
        device=args.device
    )

    # Run inference
    inference.run(
        source=args.source,
        save_output=args.save_output,
        display=not args.no_display,
        process_every=max(1, args.process_every)
    )


if __name__ == '__main__':
    main()
