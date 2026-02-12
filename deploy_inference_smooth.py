"""
deploy_inference_smooth.py  –  Previa Tech Road Audit
─────────────────────────────────────────────────────
Dual-model stakeholder demo:
  • EfficientNet tile classifier  → detects unclean / unsanitised road
  • YOLOv8 object detector        → detects potholes (filtered to ROI)

Both run on every frame.  Bounding boxes linger on screen for a
configurable duration (default 2 s) and fade out gradually.

Usage:
  python deploy_inference_smooth.py \
      --model           scripts/runs/efficientnet_224/best.pt \
      --pothole_model   smart-road-portal/ai-detection/models/pothole_yolov8s_best.pt \
      --roi             video_analysis_script/roi_configs/roi_highway_blackline.json \
      --pothole_roi     video_analysis_script/roi_configs/roi_pothole.json \
      --source          Downloads/demo.mp4 \
      --save_output     demo_smooth.mp4 \
      --no_display \
      --conf_sanitization 0.70 \
      --conf_dirty        0.80 \
      --conf_pothole      0.50 \
      --linger_seconds    2.0 \
      --device auto
"""

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


# ═══════════════════════  Detection pool  ═══════════════════════

def _detection_key(det: dict) -> tuple:
    """
    Unique key per detection.
    Includes the source model so that an EfficientNet tile at (100,200)
    and a YOLOv8 box at (100,200) are tracked independently.
    """
    x, y, w, h = det["bbox"]
    return (det.get("source", ""), x, y, w, h)


class DetectionPool:
    """
    Keeps detections alive for *linger_frames* after they were last
    seen.  Each detection fades linearly from full opacity → 0 over
    that window.

    Duplicate positions (same source + bbox) are refreshed: their
    birth frame is reset and confidence updated.
    """

    def __init__(self, linger_frames: int):
        self.linger = max(1, linger_frames)
        self._pool: dict[tuple, dict] = {}

    def update(self, new_detections: list[dict], current_frame: int):
        for det in new_detections:
            key = _detection_key(det)
            self._pool[key] = {
                "det": det,
                "birth_frame": current_frame,
            }

    def active(self, current_frame: int) -> list[tuple[dict, float]]:
        result = []
        expired = []
        for key, entry in self._pool.items():
            age = current_frame - entry["birth_frame"]
            if age >= self.linger:
                expired.append(key)
                continue
            opacity = 1.0 - (age / self.linger)
            result.append((entry["det"], opacity))
        for k in expired:
            del self._pool[k]
        return result


# ═══════════════════════  ROI helpers  ══════════════════════════

def load_roi_polygon(roi_path: str, frame_wh: tuple = None
                     ) -> np.ndarray:
    """
    Load an ROI polygon from a JSON file.  If the video resolution
    differs from the one stored in the file, the polygon is rescaled
    using the stored fractional coordinates.

    Supports two fraction formats:
      - list of [fx, fy] pairs  (pothole-style)
      - dict with bl_x/br_x/tl_x/tr_x/y_bottom/y_top  (highway-style)
    """
    with open(roi_path, "r") as f:
        data = json.load(f)
    saved_wh = tuple(data.get("frame_wh", [0, 0]))
    pts = np.array(data["roi_points"], dtype=np.int32)

    # Rescale if needed
    if (frame_wh is not None and saved_wh != (0, 0)
            and saved_wh != frame_wh):
        fracs = data.get("fractions", {})
        W, H = frame_wh
        if isinstance(fracs, dict) and "bl_x" in fracs:
            # Highway-style: dict with named fractional coordinates
            pts = np.array([
                [int(fracs["bl_x"] * W), int(fracs["y_bottom"] * H)],
                [int(fracs["br_x"] * W), int(fracs["y_bottom"] * H)],
                [int(fracs["tr_x"] * W), int(fracs["y_top"] * H)],
                [int(fracs["tl_x"] * W), int(fracs["y_top"] * H)],
            ], dtype=np.int32)
            print(f"  ROI rescaled from {saved_wh} → {frame_wh}")
        elif isinstance(fracs, list) and fracs:
            # Pothole-style: list of [fx, fy] pairs
            pts = np.array(
                [[int(fx * W), int(fy * H)] for fx, fy in fracs],
                dtype=np.int32)
            print(f"  ROI rescaled from {saved_wh} → {frame_wh}")
    return pts


def point_in_polygon(px: int, py: int, polygon: np.ndarray) -> bool:
    """Quick point-in-polygon test using OpenCV."""
    return cv2.pointPolygonTest(
        polygon.reshape((-1, 1, 2)).astype(np.float32),
        (float(px), float(py)), False) >= 0


# ═══════════════════════  YOLOv8 wrapper  ═══════════════════════

class PotholeDetector:
    """
    Thin wrapper around an ultralytics YOLOv8 model that returns
    detections in the same dict format as the EfficientNet pipeline.
    Optionally filters detections to a ROI polygon.
    """

    def __init__(self, model_path: str, confidence: float = 0.5,
                 device: str = "cpu",
                 roi_polygon: np.ndarray = None):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device
        self.roi_polygon = roi_polygon
        print(f"[Pothole] Loaded YOLOv8 from {model_path}")
        print(f"[Pothole] Confidence threshold: {confidence}")
        print(f"[Pothole] Device: {device}")
        if roi_polygon is not None:
            print(f"[Pothole] ROI filter: {len(roi_polygon)} vertices")
        else:
            print(f"[Pothole] ROI filter: none (full frame)")

    @torch.no_grad()
    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """
        Run YOLOv8 inference on a full frame.
        If a ROI polygon was provided, only detections whose centre
        falls inside the polygon are kept.
        """
        results = self.model.predict(
            frame_bgr,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )
        dets = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy   = boxes.xyxy.cpu().numpy()    # (N, 4)
                confs  = boxes.conf.cpu().numpy()    # (N,)
                clsids = boxes.cls.cpu().numpy()     # (N,)
                names  = results[0].names            # {0: 'pothole', …}

                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # ── ROI filter: keep only if centre is inside ──
                    if (self.roi_polygon is not None
                            and not point_in_polygon(
                                cx, cy, self.roi_polygon)):
                        continue

                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    cls_name = names.get(int(clsids[i]), "pothole")
                    dets.append({
                        "class":      cls_name,
                        "confidence": float(confs[i]),
                        "bbox":       (int(x1), int(y1), w, h),
                        "source":     "yolo",
                    })
        return dets


# ═══════════════════  Main inference class  ═════════════════════

def severity_color(confidence: float,
                   lo: float = 0.70, hi: float = 1.0) -> tuple:
    """
    Map a sanitization confidence score to a BGR colour that gets
    progressively more alarming:

        0.70          →  light yellow-green   (mild / semi-clean)
        0.75          →  yellow
        0.80          →  orange               (dirty threshold)
        0.90          →  red-orange
        0.95 – 1.00   →  deep red             (very dirty)

    Returns a (B, G, R) tuple with integer values 0-255.
    """
    t = max(0.0, min(1.0, (confidence - lo) / (hi - lo)))
    # Three-stop gradient: green-yellow → orange → deep-red
    if t < 0.5:
        # first half: (0,200,80) → (0,165,255) — green-yellow to orange
        s = t / 0.5
        b = int(0   * (1 - s) + 0   * s)
        g = int(200 * (1 - s) + 100 * s)
        r = int(80  * (1 - s) + 255 * s)
    else:
        # second half: (0,165,255) → (0,0,200) — orange to deep red
        s = (t - 0.5) / 0.5
        b = int(0   * (1 - s) + 0   * s)
        g = int(100 * (1 - s) + 0   * s)
        r = int(255 * (1 - s) + 200 * s)
    return (b, g, r)


class RoadAuditInference:

    # ── Static colour palette (for non-sanitization classes) ──
    COLORS = {
        "pothole":     (200, 0,   200),    # magenta  (BGR)
        "clean":       (0,   255, 0),
        "ignore":      (128, 128, 128),
        "roi":         (0,   255, 255),    # yellow
    }

    # Sanitization severity tiers
    SEMI_CLEAN_FLOOR = 0.70   # lower bound – anything below is ignored
    DIRTY_FLOOR      = 0.80   # above this → "Dirty"

    def __init__(self, model_path, roi_path,
                 conf_sanitization=0.70, conf_dirty=0.80,
                 tile_stride=112,
                 device="auto"):
        self.device_str = device
        self.device = self._pick_device(device)
        self.conf_sanitization = conf_sanitization
        self.conf_dirty = conf_dirty
        self.tile_stride = tile_stride

        # ── Load EfficientNet ──
        print(f"[Sanitization] Loading EfficientNet from {model_path} …")
        checkpoint = torch.load(model_path, map_location="cpu",
                                weights_only=False)
        self.class_names = checkpoint["class_names"]
        self.model = RoadAuditEfficientNet(
            num_classes=len(self.class_names))
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device)
        self.model.eval()
        print(f"[Sanitization] Classes: {self.class_names}")
        print(f"[Sanitization] Semi-clean: {conf_sanitization:.2f} – "
              f"{conf_dirty:.2f}")
        print(f"[Sanitization] Dirty:      ≥ {conf_dirty:.2f}")
        print(f"[Sanitization] Device: {self.device}")

        # ── ROI ──
        with open(roi_path, "r") as f:
            roi_data = json.load(f)
        self.roi_points = np.array(roi_data["roi_points"], dtype=np.int32)
        self.roi_mask = None
        self._roi_path = roi_path

        # Pothole model is attached later via attach_pothole_model()
        self.pothole: PotholeDetector | None = None
        self.pothole_roi_pts: np.ndarray | None = None

        self.fps_buffer: deque = deque(maxlen=30)

    def attach_pothole_model(self, model_path: str,
                             confidence: float = 0.5,
                             pothole_roi_path: str = ""):
        """Load YOLOv8 model and (optionally) its ROI polygon."""
        roi_poly = None
        if pothole_roi_path:
            # We'll rescale later once we know the frame size;
            # for now, load the raw points.
            roi_poly = load_roi_polygon(pothole_roi_path)
            self.pothole_roi_pts = roi_poly
            print(f"[Pothole] ROI loaded from {pothole_roi_path}")
        self.pothole = PotholeDetector(
            model_path, confidence=confidence,
            device=self.device,
            roi_polygon=roi_poly)
        self._pothole_roi_path = pothole_roi_path

    # ── device helper ──
    @staticmethod
    def _pick_device(requested: str) -> str:
        req = (requested or "auto").lower()
        if req == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if req == "mps":
            return ("mps"
                    if getattr(torch.backends, "mps", None)
                    and torch.backends.mps.is_available()
                    else "cpu")
        if req == "cpu":
            return "cpu"
        # auto
        if torch.cuda.is_available():
            return "cuda"
        if (getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()):
            return "mps"
        return "cpu"

    # ── ROI / tile helpers (for EfficientNet) ──
    def _create_roi_mask(self, shape):
        H, W = shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [self.roi_points.reshape((-1, 1, 2))], 255)
        return mask

    def _extract_tiles(self, frame, tile_size=224):
        H, W = frame.shape[:2]
        if self.roi_mask is None or self.roi_mask.shape != (H, W):
            self.roi_mask = self._create_roi_mask(frame.shape)
        x0 = int(self.roi_points[:, 0].min())
        y0 = int(self.roi_points[:, 1].min())
        x1 = int(self.roi_points[:, 0].max())
        y1 = int(self.roi_points[:, 1].max())
        tiles = []
        t = tile_size
        stride = self.tile_stride
        for y in range((y0 // stride) * stride, y1 + 1, stride):
            for x in range((x0 // stride) * stride, x1 + 1, stride):
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

    # ── EfficientNet batch inference ──
    @torch.no_grad()
    def _predict_batch(self, tiles_bgr):
        if not tiles_bgr:
            return [], [], np.zeros(
                (0, len(self.class_names)), dtype=np.float32)
        tiles_rgb = [cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
                     for t in tiles_bgr]
        tiles_pil = [Image.fromarray(t) for t in tiles_rgb]
        batch = torch.stack(
            [pil_to_tensor(p) for p in tiles_pil], dim=0
        ).to(self.device)
        logits = self.model(batch)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()
        class_idxs = probs.argmax(axis=1).astype(int).tolist()
        confs = probs.max(axis=1).astype(float).tolist()
        return class_idxs, confs, probs

    # ── Combined inference for one frame ──
    def _run_inference(self, frame) -> list[dict]:
        """Run BOTH models on a frame; return merged detection list."""
        t0 = time.time()
        all_dets: list[dict] = []

        # ── 1. EfficientNet (sanitization / unclean road) ──
        #   0.70 – 0.80  →  "semi_clean"  (mild / needs attention)
        #   ≥ 0.80       →  "dirty"       (clearly unclean)
        tiles = self._extract_tiles(frame)
        imgs = [t[0] for t in tiles]
        class_idxs, confs, _ = self._predict_batch(imgs)
        for (_, x, y), cidx, conf in zip(tiles, class_idxs, confs):
            cname = self.class_names[cidx]
            if cname == "unclean" and conf >= self.conf_sanitization:
                if conf >= self.conf_dirty:
                    severity = "dirty"
                else:
                    severity = "semi_clean"
                all_dets.append({
                    "class":      severity,
                    "confidence": conf,
                    "bbox":       (int(x), int(y), 224, 224),
                    "source":     "efficientnet",
                })

        # ── 2. YOLOv8 (pothole detection) ──
        if self.pothole is not None:
            pothole_dets = self.pothole.detect(frame)
            all_dets.extend(pothole_dets)

        elapsed = time.time() - t0
        self.fps_buffer.append(1.0 / elapsed if elapsed > 0 else 0)
        return all_dets

    # ── drawing ──
    def _get_color(self, det: dict) -> tuple:
        """
        Return BGR colour for a detection.
        Sanitization classes use a continuous confidence→colour gradient;
        other classes use the static palette.
        """
        cls = det["class"]
        if cls in ("semi_clean", "dirty"):
            return severity_color(det["confidence"],
                                  lo=self.conf_sanitization, hi=1.0)
        return self.COLORS.get(cls, (0, 165, 255))

    def _draw_annotations(self, frame, active_dets, frame_idx,
                          total_frames):
        out = frame.copy()

        # 1a. Sanitization ROI boundary (yellow)
        cv2.polylines(out,
                      [self.roi_points.reshape((-1, 1, 2))],
                      True, self.COLORS["roi"], 2)

        # 1b. Pothole ROI boundary (green) with semi-transparent fill
        if self.pothole_roi_pts is not None:
            p_roi = self.pothole_roi_pts.reshape((-1, 1, 2))
            roi_overlay = out.copy()
            cv2.fillPoly(roi_overlay, [p_roi], (0, 255, 0))
            cv2.addWeighted(roi_overlay, 0.08, out, 0.92, 0, out)
            cv2.polylines(out, [p_roi], True, (0, 255, 0), 2)

        # 2. Fading bounding boxes with severity-based colour
        for det, opacity in active_dets:
            x, y, w, h = det["bbox"]
            base_color = np.array(self._get_color(det),
                                  dtype=np.float64)

            # semi-transparent fill — heavier fill for higher severity
            overlay = out.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h),
                          base_color.tolist(), -1)
            # Dirty tiles get a stronger fill than semi-clean ones
            conf = det.get("confidence", 0.7)
            fill_strength = 0.12 + 0.14 * min(1.0, (conf - 0.7) / 0.3)
            alpha_fill = fill_strength * opacity
            cv2.addWeighted(overlay, alpha_fill, out,
                            1 - alpha_fill, 0, out)

            # solid border — thicker for higher confidence
            border = (base_color * opacity).astype(int).tolist()
            thickness = max(2, int(2 + 2 * min(1.0, (conf - 0.7) / 0.3)
                                   * opacity))
            cv2.rectangle(out, (x, y), (x + w, y + h),
                          border, thickness)

            # label text
            display_name = det["class"].replace("_", " ").title()
            label = f"{display_name}: {det['confidence']:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs = 0.55
            (tw, th_txt), _ = cv2.getTextSize(label, font, fs, 2)

            # label background
            lbl_ov = out.copy()
            cv2.rectangle(lbl_ov,
                          (x, y - th_txt - 10),
                          (x + tw + 6, y),
                          base_color.tolist(), -1)
            cv2.addWeighted(lbl_ov, opacity, out,
                            1 - opacity, 0, out)

            text_col = [int(255 * opacity)] * 3
            cv2.putText(out, label, (x + 3, y - 5),
                        font, fs, text_col, 2)

        # 3. HUD  ──  separate counts by severity
        n_dirty = sum(1 for d, _ in active_dets
                      if d.get("class") == "dirty")
        n_semi  = sum(1 for d, _ in active_dets
                      if d.get("class") == "semi_clean")
        n_pothole = sum(1 for d, _ in active_dets
                        if d.get("source") == "yolo")

        hud_y = 35

        # Dirty count (deep red)
        if n_dirty > 0:
            cv2.putText(out,
                        f"Dirty: {n_dirty}",
                        (20, hud_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 200), 2)
            hud_y += 28

        # Semi-clean count (orange)
        if n_semi > 0:
            cv2.putText(out,
                        f"Semi Clean: {n_semi}",
                        (20, hud_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 130, 255), 2)
            hud_y += 28

        # All-clear
        if n_dirty == 0 and n_semi == 0:
            cv2.putText(out,
                        "Road clean",
                        (20, hud_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 200, 0), 2)
            hud_y += 28

        # Pothole count
        if self.pothole is not None:
            if n_pothole > 0:
                cv2.putText(out,
                            f"Potholes: {n_pothole}",
                            (20, hud_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            self.COLORS["pothole"], 2)
            else:
                cv2.putText(out,
                            "No potholes",
                            (20, hud_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 200, 0), 2)
            hud_y += 28

        # Model FPS
        avg_fps = (np.mean(self.fps_buffer) if self.fps_buffer else 0)
        cv2.putText(out,
                    f"Model FPS: {avg_fps:.1f}",
                    (20, hud_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

        # 4. Branding watermark (top-right) — disabled
        # brand = "Previa Tech - Road Audit AI"
        # (bw, bh), _ = cv2.getTextSize(
        #     brand, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # bx = out.shape[1] - bw - 15
        # cv2.putText(out, brand, (bx, 25),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             (255, 255, 255), 1, cv2.LINE_AA)

        # 5. Progress bar
        if total_frames > 0:
            bar_h = 4
            progress = frame_idx / total_frames
            bar_w = int(out.shape[1] * progress)
            cv2.rectangle(out,
                          (0, out.shape[0] - bar_h),
                          (bar_w, out.shape[0]),
                          (0, 200, 255), -1)

        return out

    # ────────────────── main run loop ──────────────────

    def run(self, source, save_output=None, display=True,
            process_every=1, linger_seconds=2.0):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {source}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        out_fps = src_fps

        print(f"\nSource : {width}x{height} @ {src_fps:.1f} FPS  "
              f"({total_frames} frames)")
        print(f"Output : {out_fps:.1f} FPS  (normal speed)")
        print(f"Linger : {linger_seconds:.1f} s  "
              f"(≈ {int(linger_seconds * out_fps)} frames)\n")

        # ── Rescale sanitization ROI to actual video resolution ──
        rescaled_san = load_roi_polygon(
            self._roi_path, frame_wh=(width, height))
        if not np.array_equal(rescaled_san, self.roi_points):
            self.roi_points = rescaled_san
            self.roi_mask = None  # force mask regeneration

        # ── Rescale pothole ROI to actual video resolution ──
        if (self.pothole is not None
                and hasattr(self, "_pothole_roi_path")
                and self._pothole_roi_path):
            rescaled = load_roi_polygon(
                self._pothole_roi_path,
                frame_wh=(width, height))
            self.pothole_roi_pts = rescaled
            self.pothole.roi_polygon = rescaled

        linger_frames = max(1, int(linger_seconds * out_fps))
        pool = DetectionPool(linger_frames)

        writer = None
        if save_output:
            # Try H.264 first for smoother playback, fall back to mp4v
            for codec in ("avc1", "mp4v"):
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(save_output, fourcc,
                                         out_fps, (width, height))
                if writer.isOpened():
                    print(f"Saving → {save_output}  (codec: {codec})")
                    break
                writer.release()

        frame_idx = 0
        total_sanitization = 0
        total_potholes = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if process_every <= 1 or (frame_idx % process_every == 0):
                    dets = self._run_inference(frame)
                    pool.update(dets, frame_idx)

                    n_dirty = sum(1 for d in dets
                                  if d.get("class") == "dirty")
                    n_semi  = sum(1 for d in dets
                                  if d.get("class") == "semi_clean")
                    n_p = sum(1 for d in dets
                              if d.get("source") == "yolo")
                    total_sanitization += n_dirty + n_semi
                    total_potholes += n_p

                    if n_dirty or n_semi or n_p:
                        parts = []
                        if n_dirty:
                            parts.append(f"{n_dirty} dirty")
                        if n_semi:
                            parts.append(f"{n_semi} semi-clean")
                        if n_p:
                            parts.append(f"{n_p} pothole(s)")
                        print(f"  frame {frame_idx:5d} : "
                              + ", ".join(parts))

                active = pool.active(frame_idx)
                annotated = self._draw_annotations(
                    frame, active, frame_idx, total_frames)

                if writer:
                    writer.write(annotated)

                if display:
                    cv2.imshow("Road Audit – Smooth Demo", annotated)
                    delay = max(1, int(1000 / src_fps))
                    if (cv2.waitKey(delay) & 0xFF) == ord("q"):
                        print("Quit requested.")
                        break

                frame_idx += 1

        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

            avg = (np.mean(self.fps_buffer)
                   if self.fps_buffer else 0)
            print(f"\n{'=' * 60}")
            print("INFERENCE COMPLETE")
            print(f"{'=' * 60}")
            print(f"Frames written         : {frame_idx}")
            print(f"Unclean detections     : {total_sanitization}")
            print(f"Pothole detections     : {total_potholes}")
            print(f"Avg combined model FPS : {avg:.1f}")
            if frame_idx > 0:
                print(f"Sanitization rate      : "
                      f"{total_sanitization / frame_idx:.2f} / frame")
                print(f"Pothole rate           : "
                      f"{total_potholes / frame_idx:.2f} / frame")
            print(f"{'=' * 60}")


# ═══════════════════════════  CLI  ══════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Previa Tech Road Audit – dual-model smooth demo")

    # ── models ──
    ap.add_argument("--model", required=True,
                    help="EfficientNet sanitization model (.pt)")
    ap.add_argument("--pothole_model", type=str, default="",
                    help="YOLOv8 pothole model (.pt).  "
                         "Leave empty to skip pothole detection.")
    ap.add_argument("--roi", required=True,
                    help="ROI JSON config for sanitization tiles")
    ap.add_argument("--pothole_roi", type=str, default="",
                    help="ROI JSON for pothole detection zone.  "
                         "Only YOLOv8 boxes whose centre falls "
                         "inside this polygon are kept.")

    # ── thresholds ──
    ap.add_argument("--conf_sanitization", type=float, default=0.70,
                    help="Min confidence to flag a tile at all "
                         "(default 0.70 = semi-clean floor)")
    ap.add_argument("--conf_dirty", type=float, default=0.80,
                    help="Confidence above which a tile is 'dirty' "
                         "(default 0.80). Below this but above "
                         "--conf_sanitization → 'semi clean'.")
    ap.add_argument("--conf_pothole", type=float, default=0.50,
                    help="Confidence for pothole boxes (default 0.50)")

    # ── tile coverage ──
    ap.add_argument("--tile_stride", type=int, default=112,
                    help="Stride between sanitization tiles in pixels. "
                         "Smaller = denser coverage but slower. "
                         "Default: 112 (50%% overlap with 224px tiles)")

    # ── I/O ──
    ap.add_argument("--source", required=True,
                    help="Input video file")
    ap.add_argument("--save_output", type=str,
                    help="Output video path")
    ap.add_argument("--no_display", action="store_true",
                    help="Headless mode (no GUI window)")

    # ── performance / visual ──
    ap.add_argument("--process_every", type=int, default=1,
                    help="Run inference every Nth frame (default 1)")
    ap.add_argument("--linger_seconds", type=float, default=2.0,
                    help="Annotation linger time in seconds (default 2.0)")
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cuda", "mps", "cpu"],
                    help="Compute device (default: auto)")
    args = ap.parse_args()

    # ── Build inference engine ──
    inference = RoadAuditInference(
        model_path=args.model,
        roi_path=args.roi,
        conf_sanitization=args.conf_sanitization,
        conf_dirty=args.conf_dirty,
        tile_stride=args.tile_stride,
        device=args.device,
    )

    # ── Optionally attach pothole model ──
    if args.pothole_model:
        inference.attach_pothole_model(
            model_path=args.pothole_model,
            confidence=args.conf_pothole,
            pothole_roi_path=args.pothole_roi,
        )

    # ── Run ──
    inference.run(
        source=args.source,
        save_output=args.save_output,
        display=not args.no_display,
        process_every=max(1, args.process_every),
        linger_seconds=args.linger_seconds,
    )


if __name__ == "__main__":
    main()
