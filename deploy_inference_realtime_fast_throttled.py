import argparse
import json
import time
from pathlib import Path
from collections import deque
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.append(str(Path(__file__).parent.parent / "training"))
from scripts.model_efficientnet import RoadAuditEfficientNet
from scripts.data_utils import pil_to_tensor


class RoadAuditInference:
    
    def __init__(self, model_path, roi_path, confidence_threshold=0.7, device='cuda'):
        self.device = self._pick_device(device)
        self.confidence_threshold = confidence_threshold
        
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.class_names = checkpoint['class_names']
        
        self.model = RoadAuditEfficientNet(num_classes=len(self.class_names))
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded. Classes: {self.class_names}")
        print(f"Device: {self.device}")
        
        with open(roi_path, 'r') as f:
            roi_data = json.load(f)
        self.roi_points = np.array(roi_data['roi_points'], dtype=np.int32)
        
        self.roi_mask = None
        
        self.fps_buffer = deque(maxlen=30)
        
        self.detections = []
        
        self.colors = {
            'clean': (0, 255, 0),      
            'unclean': (0, 0, 255),    
            'ignore': (128, 128, 128), 
            'roi': (0, 255, 255)       
        }
    

    def _pick_device(self, requested: str) -> str:
        req = (requested or "auto").lower()
        if req == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if req == "mps":
            return "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        if req == "cpu":
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def create_roi_mask(self, frame_shape):
        H, W = frame_shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [self.roi_points.reshape((-1, 1, 2))], 255)
        return mask
    
    def extract_tiles(self, frame, tile_size=224):
        H, W = frame.shape[:2]
        
        if self.roi_mask is None or self.roi_mask.shape != (H, W):
            self.roi_mask = self.create_roi_mask(frame.shape)
        
        x0, y0 = int(self.roi_points[:, 0].min()), int(self.roi_points[:, 1].min())
        x1, y1 = int(self.roi_points[:, 0].max()), int(self.roi_points[:, 1].max())
        
        tiles = []
        t = tile_size
        
        for y in range((y0 // t) * t, y1 + 1, t):
            for x in range((x0 // t) * t, x1 + 1, t):
                cx, cy = x + t // 2, y + t // 2
                if not (0 <= cx < W and 0 <= cy < H):
                    continue
                if self.roi_mask[cy, cx] == 0:
                    continue
                
                x2, y2 = x + t, y + t
                if x < 0 or y < 0 or x2 > W or y2 > H:
                    continue
                
                tile_img = frame[y:y2, x:x2]
                if tile_img.shape[0] != t or tile_img.shape[1] != t:
                    continue
                
                tiles.append((tile_img, x, y))
        
        return tiles

    @torch.no_grad()
    def predict_tiles_batch(self, tiles_bgr):
        if not tiles_bgr:
            return [], [], np.zeros((0, len(self.class_names)), dtype=np.float32)

        tiles_rgb = [cv2.cvtColor(t, cv2.COLOR_BGR2RGB) for t in tiles_bgr]
        tiles_pil = [Image.fromarray(t) for t in tiles_rgb]
        batch = torch.stack([pil_to_tensor(p) for p in tiles_pil], dim=0).to(self.device)

        logits = self.model(batch)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()

        class_idxs = probs.argmax(axis=1).astype(int).tolist()
        confidences = probs.max(axis=1).astype(float).tolist()
        return class_idxs, confidences, probs

    @torch.no_grad()
    def predict_tile(self, tile_bgr):
        tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
        
        tile_pil = Image.fromarray(tile_rgb)
        
        tile_tensor = pil_to_tensor(tile_pil).unsqueeze(0).to(self.device)
        
        logits = self.model(tile_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        
        class_idx = int(probs.argmax())
        confidence = float(probs[class_idx])
        
        return class_idx, confidence, probs
    
    def process_frame(self, frame):
        start_time = time.time()

        tiles = self.extract_tiles(frame)
        detections_this_frame = []

        tile_imgs = [t[0] for t in tiles]
        class_idxs, confidences, _ = self.predict_tiles_batch(tile_imgs)

        for (tile_img, x, y), class_idx, conf in zip(tiles, class_idxs, confidences):
            class_name = self.class_names[class_idx]
            if class_name == 'unclean' and conf >= self.confidence_threshold:
                detections_this_frame.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': (int(x), int(y), 224, 224),
                    'timestamp': time.time()
                })

        annotated = self.annotate_frame(frame, detections_this_frame)

        elapsed = time.time() - start_time
        self.fps_buffer.append(1.0 / elapsed if elapsed > 0 else 0)
        return annotated, detections_this_frame

    def annotate_frame(self, frame, detections):
        annotated = frame.copy()
        
        cv2.polylines(annotated, [self.roi_points.reshape((-1, 1, 2))], 
                     True, self.colors['roi'], 2)
        
        for det in detections:
            x, y, w, h = det['bbox']
            color = self.colors[det['class']]
            
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
            
            label = f"{det['class']}: {det['confidence']:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            cv2.putText(annotated, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        fps = np.mean(self.fps_buffer) if len(self.fps_buffer) > 0 else 0
        cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        cv2.putText(annotated, f"Unclean tiles: {len(detections)}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return annotated
    
    def run(self, source, save_output=None, display=True, process_every: int = 1):
        if isinstance(source, int) or source.isdigit():
            cap = cv2.VideoCapture(int(source))
            print(f"Opened camera {source}")
        else:
            cap = cv2.VideoCapture(source)
            print(f"Opened video file: {source}")
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {width}×{height} @ {fps:.1f} FPS")
        
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_fps = args.output_fps if args.output_fps and args.output_fps > 0 else fps
            out = cv2.VideoWriter(save_output, fourcc, out_fps, (width, height))
            print(f"Saving output to: {save_output}")
        
        frame_count = 0
        total_detections = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if process_every <= 1 or (frame_count % process_every == 0):
                    annotated, detections = self.process_frame(frame)
                else:
                    detections = []
                    annotated = self.annotate_frame(frame, detections)
                
                total_detections += len(detections)
                if len(detections) > 0:
                    print(f"Frame {frame_count}: {len(detections)} unclean tiles detected")
                
                if writer:
                    writer.write(annotated)
                
                delay_ms = 1
                if args.playback_fps and args.playback_fps > 0:
                    delay_ms = max(1, int(1000.0 / args.playback_fps))
                if display:
                    cv2.imshow('Road Audit - Real-time Inference', annotated)
                    key = cv2.waitKey(delay_ms) & 0xFF
                    if key == ord('q'):
                        print("Quit requested")
                        break
                    elif key == ord('s'):
                        screenshot_path = f"screenshot_{frame_count:06d}.jpg"
                        cv2.imwrite(screenshot_path, annotated)
                        print(f"Saved screenshot: {screenshot_path}")
                
                frame_count += 1
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            
            avg_fps = np.mean(self.fps_buffer) if len(self.fps_buffer) > 0 else 0
            print(f"\n{'='*60}")
            print("INFERENCE COMPLETE")
            print("="*60)
            print(f"Frames processed:    {frame_count}")
            print(f"Total detections:    {total_detections}")
            print(f"Average FPS:         {avg_fps:.1f}")
            print(f"Detection rate:      {total_detections/frame_count:.2f} per frame")
            print("="*60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to trained model (.pt file)")
    ap.add_argument("--roi", required=True, help="Path to ROI JSON config")
    ap.add_argument("--source", required=True, help="Video file or camera index (0 for webcam)")
    ap.add_argument("--confidence", type=float, default=0.7, 
                   help="Confidence threshold for detections (default: 0.7)")
    ap.add_argument("--save_output", type=str, help="Path to save annotated video")
    ap.add_argument("--no_display", action="store_true", help="Don't display video window")
    ap.add_argument("--process_every", type=int, default=1, help="Run inference every Nth frame (default: 1)")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"],
                   help="Device for inference: auto/cuda/mps/cpu (default: auto)")
    args = ap.parse_args()
    
    inference = RoadAuditInference(
        model_path=args.model,
        roi_path=args.roi,
        confidence_threshold=args.confidence,
        device=args.device
    )
    
    inference.run(
        source=args.source,
        save_output=args.save_output,
        process_every=max(1, int(args.process_every)),
        display=not args.no_display
    )


if __name__ == "__main__":
    main()