"""
make_roi_auto.py
Auto-generate trapezoid ROI for road audit (highway vs sector).

Output format:
{
  "roi_points": [[x_bl,y_bl],[x_br,y_br],[x_tr,y_tr],[x_tl,y_tl]],
  "mode": "highway"|"sector",
  "frame_wh": [W,H],
  "fractions": {...}
}

Usage:
  python3 make_roi_auto.py /path/to/video.mp4 --mode highway --out roi_highway.json
  python3 make_roi_auto.py /path/to/video.mp4 --mode sector  --out roi_sector.json

Deps:
  pip install opencv-python
"""
import argparse, json
from pathlib import Path
import cv2

PRESETS = {
    "highway": dict(
        bl_x=0.02, br_x=0.98,     
        y_bottom=0.92,            
        tl_x=0.20, tr_x=0.80,     
        y_top=0.52       
    ),
    "sector": dict(
        bl_x=0.02, br_x=0.98,
        y_bottom=0.92,
        tl_x=0.20, tr_x=0.80,
        y_top=0.52
    )
}

def read_wh(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if W <= 0 or H <= 0:
        raise SystemExit(f"Could not read resolution for: {video_path}")
    return W, H

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def mk_points(W, H, frac):
    x_bl = int(round(frac["bl_x"] * W))
    x_br = int(round(frac["br_x"] * W))
    y_bl = int(round(frac["y_bottom"] * H))
    y_br = y_bl
    x_tr = int(round(frac["tr_x"] * W))
    x_tl = int(round(frac["tl_x"] * W))
    y_top = int(round(frac["y_top"] * H))

    pts = [
        [clamp(x_bl, 0, W-1), clamp(y_bl, 0, H-1)],
        [clamp(x_br, 0, W-1), clamp(y_br, 0, H-1)],
        [clamp(x_tr, 0, W-1), clamp(y_top, 0, H-1)],
        [clamp(x_tl, 0, W-1), clamp(y_top, 0, H-1)],
    ]
    return pts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=str)
    ap.add_argument("--mode", choices=["highway","sector"], required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    W, H = read_wh(args.video)
    frac = PRESETS[args.mode]
    pts = mk_points(W, H, frac)

    payload = {
        "roi_points": pts,
        "mode": args.mode,
        "frame_wh": [W, H],
        "fractions": frac
    }
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.out}  (W,H)=({W},{H})  pts={pts}")

if __name__ == "__main__":
    main()