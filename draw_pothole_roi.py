"""
draw_pothole_roi.py  –  Interactive ROI editor for pothole detection
────────────────────────────────────────────────────────────────────
Opens a video frame and lets you drag polygon vertices to match
the road surface boundary. Saves the result as a JSON config.

Controls:
  • Left-click near a vertex  → select it (turns yellow)
  • Drag                      → move the selected vertex
  • Right-click               → add a new vertex after the selected one
  • 'd'                       → delete the selected vertex (min 3)
  • 'r'                       → reset to the initial polygon
  • 's'                       → save the ROI to the output JSON file
  • 'q' / ESC                 → quit (auto-saves)

Usage:
  python draw_pothole_roi.py \
      --source /Users/shubhamfufal/Downloads/demo.mp4 \
      --output video_analysis_script/roi_configs/roi_pothole.json \
      --frame 50
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path

# ── globals for mouse callback ──
points = []
selected_idx = -1
dragging = False
SNAP_RADIUS = 15  # pixels


def load_existing_roi(path):
    """Load points from an existing ROI JSON, if it exists."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r") as f:
        data = json.load(f)
    return [list(pt) for pt in data["roi_points"]]


def save_roi(path, pts, frame_wh):
    """Write the ROI polygon to JSON."""
    W, H = frame_wh
    fractions = [[round(x / W, 4), round(y / H, 4)] for x, y in pts]
    data = {
        "roi_points": pts,
        "mode": "pothole",
        "frame_wh": list(frame_wh),
        "fractions": fractions,
        "notes": "Pothole ROI – edited with draw_pothole_roi.py",
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved ROI ({len(pts)} points) → {path}")


def draw(frame, pts, sel):
    """Draw the polygon and vertices on the frame."""
    vis = frame.copy()
    n = len(pts)
    if n == 0:
        return vis

    np_pts = np.array(pts, dtype=np.int32)

    # filled semi-transparent overlay
    overlay = vis.copy()
    cv2.fillPoly(overlay, [np_pts.reshape((-1, 1, 2))], (0, 255, 0))
    cv2.addWeighted(overlay, 0.20, vis, 0.80, 0, vis)

    # polygon outline
    cv2.polylines(vis, [np_pts.reshape((-1, 1, 2))], True,
                  (0, 255, 0), 2)

    # vertices
    for i, (x, y) in enumerate(pts):
        color = (0, 255, 255) if i == sel else (255, 255, 255)
        cv2.circle(vis, (x, y), 6, color, -1)
        cv2.circle(vis, (x, y), 6, (0, 0, 0), 1)
        cv2.putText(vis, str(i), (x + 8, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # instructions
    lines = [
        "Click vertex to select, drag to move",
        "Right-click: add point | 'd': delete | 'r': reset",
        "'s': save | 'q'/ESC: quit (auto-saves)",
    ]
    for i, txt in enumerate(lines):
        cv2.putText(vis, txt, (10, 20 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
        cv2.putText(vis, txt, (10, 20 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return vis


def mouse_cb(event, x, y, flags, param):
    global points, selected_idx, dragging

    if event == cv2.EVENT_LBUTTONDOWN:
        # find nearest vertex
        best_i, best_d = -1, 1e9
        for i, (px, py) in enumerate(points):
            d = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
            if d < best_d:
                best_i, best_d = i, d
        if best_d < SNAP_RADIUS:
            selected_idx = best_i
            dragging = True
        else:
            selected_idx = -1

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging and selected_idx >= 0:
            points[selected_idx] = [x, y]

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

    elif event == cv2.EVENT_RBUTTONDOWN:
        # add a new point after the selected one
        ins = (selected_idx + 1) if selected_idx >= 0 else len(points)
        points.insert(ins, [x, y])
        selected_idx = ins


def main():
    global points, selected_idx

    ap = argparse.ArgumentParser(
        description="Interactive pothole ROI editor")
    ap.add_argument("--source", required=True,
                    help="Video file to grab a frame from")
    ap.add_argument("--output", type=str,
                    default="video_analysis_script/roi_configs/"
                            "roi_pothole.json",
                    help="Output ROI JSON path")
    ap.add_argument("--frame", type=int, default=50,
                    help="Frame index to display (default 50)")
    args = ap.parse_args()

    # grab a frame
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {args.source}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {args.frame}")
    print(f"Frame {args.frame}: {W}x{H}")

    # load existing ROI or create default trapezoid
    existing = load_existing_roi(args.output)
    if existing:
        # Scale points if frame resolution differs from saved config
        with open(args.output, "r") as f:
            saved = json.load(f)
        saved_wh = saved.get("frame_wh", [W, H])
        if saved_wh[0] != W or saved_wh[1] != H:
            # use fractions to recompute for current resolution
            fracs = saved.get("fractions", [])
            if fracs:
                points = [[int(fx * W), int(fy * H)]
                          for fx, fy in fracs]
                print(f"Scaled ROI from {saved_wh} → {W}x{H}")
            else:
                points = existing
        else:
            points = existing
        print(f"Loaded {len(points)} points from {args.output}")
    else:
        # default: trapezoid covering bottom 40% of frame
        points = [
            [int(0.02 * W), int(0.97 * H)],
            [int(0.04 * W), int(0.70 * H)],
            [int(0.06 * W), int(0.65 * H)],
            [int(0.50 * W), int(0.65 * H)],
            [int(0.90 * W), int(0.63 * H)],
            [int(0.95 * W), int(0.70 * H)],
            [int(0.98 * W), int(0.97 * H)],
        ]
        print("Created default ROI polygon")

    initial_points = [pt[:] for pt in points]  # for reset

    win = "Pothole ROI Editor"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(W, 1200), min(H, 800))
    cv2.setMouseCallback(win, mouse_cb)

    while True:
        vis = draw(frame, points, selected_idx)
        cv2.imshow(win, vis)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), 27):  # quit
            save_roi(args.output, points, (W, H))
            break
        elif key == ord("s"):      # save
            save_roi(args.output, points, (W, H))
        elif key == ord("r"):      # reset
            points = [pt[:] for pt in initial_points]
            selected_idx = -1
            print("Reset to initial polygon")
        elif key == ord("d"):      # delete selected
            if selected_idx >= 0 and len(points) > 3:
                del points[selected_idx]
                selected_idx = min(selected_idx, len(points) - 1)
                print(f"Deleted vertex. {len(points)} remain.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
