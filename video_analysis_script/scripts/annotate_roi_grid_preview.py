import argparse, json
import cv2
import numpy as np

def load_roi(roi_json):
    with open(roi_json, "r") as f:
        d = json.load(f)
    pts = np.array(d["roi_points"], dtype=np.int32)
    if pts.shape != (4,2):
        raise ValueError("roi_points must be 4x2")
    return pts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("roi")
    ap.add_argument("out")
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--max_w", type=int, default=3840, help="Max width for preview (default: 3840 for 4K, supports 2K natively)")
    ap.add_argument("--max_frames", type=int, default=0, help="0 = all frames")
    args = ap.parse_args()

    roi_pts = load_roi(args.roi)

    cap = cv2.VideoCapture(args.inp)
    if not cap.isOpened():
        raise SystemExit("Could not open input video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = 1.0
    if W > args.max_w:
        scale = args.max_w / W
    W2, H2 = int(W*scale), int(H*scale)
    roi = (roi_pts.astype(np.float32) * scale).astype(np.int32)

    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W2, H2))

    mask = np.zeros((H2, W2), dtype=np.uint8)
    cv2.fillPoly(mask, [roi.reshape((-1,1,2))], 255)

    t = args.tile
    x0, x1 = int(roi[:,0].min()), int(roi[:,0].max())
    y0, y1 = int(roi[:,1].min()), int(roi[:,1].max())

    frame_i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if scale != 1.0:
            fr = cv2.resize(fr, (W2, H2))

        cv2.polylines(fr, [roi.reshape((-1,1,2))], True, (0,255,255), 2)

        for y in range((y0//t)*t, y1+1, t):
            for x in range((x0//t)*t, x1+1, t):
                cx, cy = x + t//2, y + t//2
                if 0 <= cx < W2 and 0 <= cy < H2 and mask[cy, cx] > 0:
                    cv2.rectangle(fr, (x, y), (x+t, y+t), (0,255,255), 1)

        cv2.putText(fr, "ROI+Grid preview (no detection)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        out.write(fr)
        frame_i += 1
        if args.max_frames and frame_i >= args.max_frames:
            break

    cap.release()
    out.release()
    print(f"Wrote {args.out} - Resolution: {W2}x{H2} (scale: {scale:.2f}x)")

if __name__ == "__main__":
    main()