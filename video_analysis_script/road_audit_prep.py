"""
Label keys in GUI:
  0 = clean_road
  1 = unclean_road
  2 = ignore
  a/d = prev/next tile
  s = save
  q = quit
"""

import argparse
import csv
import json
import os
from pathlib import Path
from dataclasses import dataclass
import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}

LABEL_MAP = {
    0: "clean_road",
    1: "unclean_road",
    2: "ignore",
}

@dataclass
class ROIConfig:
    mode: str
    pts: np.ndarray
    frame_wh: tuple

def load_roi(roi_path: str) -> ROIConfig:
    with open(roi_path, "r") as f:
        d = json.load(f)
    pts = np.array(d["roi_points"], dtype=np.int32)
    if pts.shape != (4, 2):
        raise ValueError(f"Bad roi_points in {roi_path}")
    mode = d.get("mode", "unknown")
    frame_wh = tuple(d.get("frame_wh", [0, 0]))
    return ROIConfig(mode=mode, pts=pts, frame_wh=frame_wh)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def infer_category(video_path: Path) -> str:
    s = str(video_path).lower()
    if "sector" in s:
        return "sector"
    return "highway"

def get_resolution(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if W <= 0 or H <= 0:
        return None
    return W, H, fps, n_frames

def sample_frame_indices(n_frames: int, src_fps: float, target_fps: float):
    if n_frames <= 0 or src_fps <= 0 or target_fps <= 0:
        return []
    step = max(1, int(round(src_fps / target_fps)))
    return list(range(0, n_frames, step))

def polygon_mask(H, W, pts):
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.reshape((-1, 1, 2))], 255)
    return mask

def overlay_roi_grid(frame, roi_pts, tile, color=(0,255,255)):
    out = frame.copy()
    cv2.polylines(out, [roi_pts.reshape((-1,1,2))], True, color, 2)

    H, W = out.shape[:2]
    mask = polygon_mask(H, W, roi_pts)
    x0, x1 = int(roi_pts[:,0].min()), int(roi_pts[:,0].max())
    y0, y1 = int(roi_pts[:,1].min()), int(roi_pts[:,1].max())

    t = tile
    for y in range((y0//t)*t, y1+1, t):
        for x in range((x0//t)*t, x1+1, t):
            cx, cy = x + t//2, y + t//2
            if 0 <= cx < W and 0 <= cy < H and mask[cy, cx] > 0:
                cv2.rectangle(out, (x,y), (x+t,y+t), color, 1)

    return out

def iter_videos(in_dir: Path):
    for p in sorted(in_dir.rglob("*")):
        if p.suffix.lower() in VIDEO_EXTS:
            yield p

def select_roi_for_video(video_path: Path, W: int, H: int, args) -> str:
    cat = infer_category(video_path)
    if cat == "highway":
        if (W, H) == (848, 480):
            return args.roi_highway_848
        if (W, H) == (832, 464):
            return args.roi_highway_832
        return args.roi_highway_832 or args.roi_highway_848
    else:
        if args.sector_mode == "strict":
            return args.roi_sector_strict
        return args.roi_sector_wide

def extract_tiles_from_frame(frame_bgr, roi_pts, tile, out_tiles_dir: Path,
                             meta_rows: list, video_id: str, frame_idx: int,
                             quality: int = 98, fmt: str = "jpg"):
    H, W = frame_bgr.shape[:2]
    mask = polygon_mask(H, W, roi_pts)

    x0, x1 = int(roi_pts[:,0].min()), int(roi_pts[:,0].max())
    y0, y1 = int(roi_pts[:,1].min()), int(roi_pts[:,1].max())

    t = tile
    for y in range((y0//t)*t, y1+1, t):
        for x in range((x0//t)*t, x1+1, t):
            cx, cy = x + t//2, y + t//2
            if not (0 <= cx < W and 0 <= cy < H):
                continue
            if mask[cy, cx] == 0:
                continue

            x2, y2 = x + t, y + t
            if x < 0 or y < 0 or x2 > W or y2 > H:
                continue

            tile_img = frame_bgr[y:y2, x:x2]
            ext = "png" if fmt == "png" else "jpg"
            tile_name = f"{video_id}__f{frame_idx:06d}__x{x:04d}_y{y:04d}.{ext}"
            out_path = out_tiles_dir / tile_name

            if fmt == "png":
                cv2.imwrite(str(out_path), tile_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            else:
                cv2.imwrite(str(out_path), tile_img, [cv2.IMWRITE_JPEG_QUALITY, quality])

            meta_rows.append({
                "tile_path": str(out_path),
                "label_id": "",
                "label_name": "",
                "video_id": video_id,
                "frame_idx": frame_idx,
                "x": x,
                "y": y,
                "tile": t,
            })

def write_csv(csv_path: Path, rows: list):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def read_csv(csv_path: Path):
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        return list(r)

def save_csv(csv_path: Path, rows: list):
    if not rows:
        return
    fields = rows[0].keys()
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)

def label_gui(labels_csv: Path):
    rows = read_csv(labels_csv)
    if not rows:
        print("No rows in labels.csv")
        return

    i = 0
    win = "Label Tiles (0 clean, 1 unclean, 2 ignore, a/d prev/next, s save, q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def render(idx):
        row = rows[idx]
        img = cv2.imread(row["tile_path"])
        if img is None:
            img = np.zeros((256,256,3), dtype=np.uint8)
        label = row.get("label_name","")
        info = f"{idx+1}/{len(rows)}  {Path(row['tile_path']).name}  label={label}"
        vis = img.copy()
        cv2.putText(vis, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        return vis

    while True:
        vis = render(i)
        cv2.imshow(win, vis)
        k = cv2.waitKey(0) & 0xFF

        if k in (ord('q'), 27):
            break
        if k == ord('a'):
            i = max(0, i-1)
            continue
        if k == ord('d'):
            i = min(len(rows)-1, i+1)
            continue
        if k == ord('s'):
            save_csv(labels_csv, rows)
            print(f"Saved: {labels_csv}")
            continue

        if k in (ord('0'), ord('1'), ord('2')):
            lid = int(chr(k))
            rows[i]["label_id"] = str(lid)
            rows[i]["label_name"] = LABEL_MAP[lid]
            i = min(len(rows)-1, i+1)

    save_csv(labels_csv, rows)
    cv2.destroyAllWindows()
    print(f"Final saved: {labels_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, help="Root folder containing videos")
    ap.add_argument("--out_dir", type=str, required=True, help="Output folder")
    ap.add_argument("--fps", type=float, default=5.0, help="Sampling FPS (use 5 for short clips)")
    ap.add_argument("--tile", type=int, default=224, help="Tile size in pixels (224 recommended for ML models)")
    ap.add_argument("--max_w", type=int, default=0, help="Resize width for previews/tiles (0=native resolution)")
    ap.add_argument("--quality", type=int, default=98, help="JPEG quality 1-100 (98 recommended, 100=max)")
    ap.add_argument("--format", type=str, default="jpg", choices=["jpg", "png"], help="Tile output format (png=lossless)")
    ap.add_argument("--make_previews", action="store_true", help="Write ROI+grid preview videos")
    ap.add_argument("--max_preview_frames", type=int, default=300, help="Preview frame cap")
    ap.add_argument("--label", action="store_true", help="Launch labeling GUI on out_dir/labels.csv")

    ap.add_argument("--sector_mode", choices=["strict","wide"], default="strict")

    ap.add_argument("--roi_highway_832", type=str, default="", help="ROI json for 832x464 highway")
    ap.add_argument("--roi_highway_848", type=str, default="", help="ROI json for 848x480 highway")
    ap.add_argument("--roi_sector_wide", type=str, default="", help="ROI json for sector wide")
    ap.add_argument("--roi_sector_strict", type=str, default="", help="ROI json for sector strict")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if args.label:
        labels_csv = out_dir / "labels.csv"
        if not labels_csv.exists():
            error_msg = f"labels.csv not found at {labels_csv}\n"
            error_msg += f"\nPlease ensure labels.csv exists in: {out_dir}\n"
            parent_dir = out_dir.parent
            possible_locations = [
                parent_dir / "tiles_224_extracted" / "labels.csv",
                parent_dir / "tiles_224" / "labels.csv",
                parent_dir / "labeled_subset" / "labels.csv",
            ]
            found_alternatives = [p for p in possible_locations if p.exists()]
            if found_alternatives:
                error_msg += f"\nFound labels.csv in alternative location(s):\n"
                for alt in found_alternatives:
                    error_msg += f"  - {alt}\n"
                error_msg += f"\nTry using: --out_dir {found_alternatives[0].parent}"
            raise SystemExit(error_msg)
        label_gui(labels_csv)
        return

    if not args.in_dir:
        raise SystemExit("--in_dir is required unless --label is used")
    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        raise SystemExit(f"in_dir not found: {in_dir}")

    videos = list(iter_videos(in_dir))
    total_videos = len(videos)
    if total_videos == 0:
        print(f"No videos found under: {in_dir}")
        return
    print(f"Found {total_videos} video(s) under {in_dir}")

    roi_cache = {}

    previews_dir = out_dir / "previews"
    if args.make_previews:
        ensure_dir(previews_dir)

    all_rows = []
    rows_by_group = {}
    skipped = []

    for vid_idx, vp in enumerate(videos, start=1):
        print(f"\n[{vid_idx}/{total_videos}] Processing video: {vp}")
        res = get_resolution(vp)
        if res is None:
            skipped.append((str(vp), "unreadable_resolution"))
            print(f"  Skipping (cannot read resolution).")
            continue
        W, H, src_fps, n_frames = res
        print(f"  Resolution: {W}x{H}, frames: {n_frames}, src_fps: {src_fps:.2f}")

        roi_path = select_roi_for_video(vp, W, H, args)
        if not roi_path or not Path(roi_path).exists():
            skipped.append((str(vp), f"missing_roi_for_{W}x{H}_{infer_category(vp)}"))
            print(f"  Skipping (missing ROI config for {W}x{H}, category={infer_category(vp)}).")
            continue

        if roi_path not in roi_cache:
            roi_cache[roi_path] = load_roi(roi_path)
        roi_cfg = roi_cache[roi_path]

        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            skipped.append((str(vp), "cannot_open"))
            continue

        scale = 1.0
        if args.max_w and W > args.max_w:
            scale = args.max_w / W
        W2, H2 = int(W*scale), int(H*scale)
        roi_pts = (roi_cfg.pts.astype(np.float32) * scale).astype(np.int32)

        preview_writer = None
        if args.make_previews:
            out_prev = previews_dir / f"{vp.stem}__roi_{infer_category(vp)}_{W}x{H}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            preview_writer = cv2.VideoWriter(str(out_prev), fourcc, src_fps or 15.0, (W2, H2))

        idxs = sample_frame_indices(n_frames, src_fps, args.fps)
        total_frames_for_video = len(idxs)
        print(f"  Sampled {total_frames_for_video} frame(s) for extraction.")

        group_name = vp.parent.name
        group_dir = out_dir / group_name
        group_tiles_dir = group_dir / "tiles"
        ensure_dir(group_tiles_dir)
        if group_name not in rows_by_group:
            rows_by_group[group_name] = []

        video_id = f"{group_name}_{vp.stem}"

        log_every = max(1, total_frames_for_video // 10)

        for j, frame_idx in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, fr = cap.read()
            if not ok or fr is None:
                continue
            if scale != 1.0:
                fr = cv2.resize(fr, (W2, H2))


            if preview_writer is not None and (args.max_preview_frames <= 0 or j < args.max_preview_frames):
                vis = overlay_roi_grid(fr, roi_pts, args.tile)
                cv2.putText(vis, f"{vp.name}  {infer_category(vp)}  {W}x{H}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                preview_writer.write(vis)

            frame_rows = []
            extract_tiles_from_frame(
                fr,
                roi_pts,
                args.tile,
                group_tiles_dir,
                frame_rows,
                video_id,
                frame_idx,
                quality=args.quality,
                fmt=args.format,
            )
            all_rows.extend(frame_rows)
            rows_by_group[group_name].extend(frame_rows)

            if (j + 1) % log_every == 0 or (j + 1) == total_frames_for_video:
                print(
                    f"  Frames processed for {vp.name}: {j+1}/{total_frames_for_video} "
                    f"(total tiles so far: {len(all_rows)})"
                )

        cap.release()
        if preview_writer is not None:
            preview_writer.release()


    labels_csv = out_dir / "labels.csv"
    write_csv(labels_csv, all_rows)

    for group_name, rows in rows_by_group.items():
        group_dir = out_dir / group_name
        group_labels_csv = group_dir / "labels.csv"
        write_csv(group_labels_csv, rows)

    if skipped:
        skipped_csv = out_dir / "skipped_videos.csv"
        with open(skipped_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "reason"])
            w.writerows(skipped)
        print(f"Skipped videos report: {skipped_csv}")

    print(f"\nDONE.")
    print(f"Tiles and labels written under: {out_dir}")
    print(f"Global labels:  {labels_csv}")
    if rows_by_group:
        print("Per-folder outputs:")
        for group_name in sorted(rows_by_group.keys()):
            group_dir = out_dir / group_name
            group_labels_csv = group_dir / "labels.csv"
            print(f"  - {group_name}: tiles={group_dir / 'tiles'}  labels={group_labels_csv}  "
                  f"rows={len(rows_by_group[group_name])}")
    if args.make_previews:
        print(f"Previews: {previews_dir}")
    print(f"Total tiles (all folders): {len(all_rows)}")

if __name__ == "__main__":
    main()
