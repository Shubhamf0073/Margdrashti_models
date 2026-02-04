import argparse, math
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from numpy.ma.core import append

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}

def lap_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def black_frac(gray: np.ndarray, thresh: int = 18) -> float:
    return float((gray < thresh).mean())

def sample_idxs(n_frames: int, n_samples: int):
    if n_frames <= 0: return []
    if n_frames <= n_samples: return list(range(n_frames))
    idxs = np.linspace(0, n_frames - 1, n_samples).round().astype(int).tolist()
    out, seen = [], set()
    for i in idxs:
        if i not in seen:
            out.append(i); seen.add(i)
    return out

def make_montage(frames, cols=4, max_w=1280):
    if not frames: return None
    h, w = frames[0].shape[:2]
    cols = min(cols, len(frames))
    rows = math.ceil(len(frames) / cols)
    m = np.zeros((rows*h, cols*w, 3), dtype=np.uint8)
    for i, fr in enumerate(frames):
        r, c = divmod(i, cols)
        m[r*h:(r+1)*h, c*w:(c+1)*w] = fr
    if m.shape[1] > max_w:
        s = max_w / m.shape[1]
        m = cv2.resize(m, (int(m.shape[1]*s), int(m.shape[0]*s)))
    return m

def analyze(path: Path, n_samples=12, resize_w=960):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"status":"error_open"}

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    idxs = sample_idxs(n_frames, n_samples)
    br, blur, blk = [], [], []
    sampled = []

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if not ok or fr is None:
            continue
        if resize_w and fr.shape[1] > resize_w:
            s = resize_w / fr.shape[1]
            fr = cv2.resize(fr, (resize_w, int(fr.shape[0]*s)))

        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        br.append(float(g.mean()))
        blur.append(lap_var(g))
        blk.append(black_frac(g))
        sampled.append(fr)

    cap.release()

    if not br:
        return {"status":"error_read", "frames":n_frames, "fps":fps, "w":w, "h":h}, None

    def q(x,p): return float(np.quantile(np.array(x), p))
    m = {
        "status":"ok", "frames":n_frames, "fps":fps, "w":w, "h":h,
        "brightness_mean": float(np.mean(br)),
        "brightness_p10": q(br,0.10),
        "blur_median": float(np.median(blur)),
        "blur_p10": q(blur,0.10),
        "blackfrac_median": float(np.median(blk)),
        "blackfrac_p90": q(blk,0.90),
        "samples_used": len(br),
    }

    suggest, reasons = "KEEP", []
    if m["blackfrac_p90"] > 0.65:
        suggest = "REMOVE"; reasons.append("many_black_or_occluded_frames")
    if m["brightness_p10"] < 25:
        reasons.append("very_dark_segments")
    if m["blur_p10"] < 20:
        reasons.append("very_blurry_segments")
    if m["frames"] < 30 or m["w"] == 0 or m["h"] == 0:
        suggest = "REMOVE"; reasons.append("too_short_or_invalid")

    m["suggest"] = suggest
    m["reasons"] = "|".join(reasons)
    return m, make_montage(sampled, cols=4)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir")
    ap.add_argument("--out", default="qc_out")
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--resize_w", type=int, default=960)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out)
    (out_dir / "qc_montages").mkdir(parents=True, exist_ok=True)

    videos = sorted([p for p in in_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS])
    if not videos:
        print("No videos found."); return

    rows = []
    for vp in videos:
        res, montage = analyze(vp, args.samples, args.resize_w)
        res["path"] = str(vp); res["name"] = vp.name
        rows.append(res)
        if montage is not None:
            cv2.imwrite(str(out_dir/"qc_montages"/(vp.stem+"_montage.jpg")), montage)
        print(vp.name, res.get("suggest"), res.get("reasons",""))

    qc_path = out_dir / "qc_report.csv"
    df = pd.DataFrame(rows)

    df.to_csv(
        qc_path,
        mode="a" if qc_path.exists() else "w",
        header=not qc_path.exists(),
        index=False
    )
    print("Wrote:", out_dir/"qc_report.csv")

if __name__ == "__main__":
    main()
