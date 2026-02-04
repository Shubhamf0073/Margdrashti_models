"""
transfer_labels_to_224.py

Transfer labels from your 128×128 labeled dataset to 224×224 tiles.

Strategy:
- Match tiles by video_id and frame_idx
- 128×128 tiles at position (x, y) map to same region in 224×224
- Multiple 224×224 tiles may cover same region as one 128×128 tile

Usage:
    python transfer_labels_to_224.py \
        --labels_128 data/road_audit_labeled_6792_clean/labels.csv \
        --tiles_224_dir data/tiles_224/tiles \
        --out_csv data/tiles_224/labels_transferred.csv

This creates a labeled CSV for your 224×224 tiles using your existing labels.
"""

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def parse_tile_name(tile_name):
    """
    Extract video_id, frame_idx, x, y from tile filename.
    
    Format: {video_id}__f{frame:06d}__x{x:04d}_y{y:04d}.{ext}
    Example: clean_highway_clean_highway1__f000252__x0384_y0256.jpg
    
    Returns:
        dict with keys: video_id, frame_idx, x, y
    """
    stem = Path(tile_name).stem
    
    try:
        # Split by '__'
        parts = stem.split('__')
        video_id = parts[0]
        
        # Extract frame number
        frame_part = [p for p in parts if p.startswith('f')][0]
        frame_idx = int(frame_part[1:])
        
        # Extract x, y
        xy_part = [p for p in parts if p.startswith('x')][0]
        x_str, y_str = xy_part.split('_')
        x = int(x_str[1:])
        y = int(y_str[1:])
        
        return {
            'video_id': video_id,
            'frame_idx': frame_idx,
            'x': x,
            'y': y
        }
    except Exception as e:
        print(f"Warning: Could not parse {tile_name}: {e}")
        return None


def tiles_overlap(tile1_x, tile1_y, tile1_size, tile2_x, tile2_y, tile2_size, threshold=0.5):
    """
    Check if two tiles overlap by at least threshold fraction.
    
    Args:
        tile1_x, tile1_y: top-left corner of tile 1
        tile1_size: size of tile 1
        tile2_x, tile2_y: top-left corner of tile 2  
        tile2_size: size of tile 2
        threshold: minimum overlap fraction (0-1)
    
    Returns:
        bool: True if overlap > threshold
    """
    # Calculate overlap rectangle
    x1 = max(tile1_x, tile2_x)
    y1 = max(tile1_y, tile2_y)
    x2 = min(tile1_x + tile1_size, tile2_x + tile2_size)
    y2 = min(tile1_y + tile1_size, tile2_y + tile2_size)
    
    # No overlap
    if x2 <= x1 or y2 <= y1:
        return False
    
    # Calculate overlap area
    overlap_area = (x2 - x1) * (y2 - y1)
    
    # Calculate smaller tile area (more conservative)
    smaller_area = min(tile1_size ** 2, tile2_size ** 2)
    
    # Check if overlap meets threshold
    overlap_fraction = overlap_area / smaller_area
    return overlap_fraction >= threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_128", required=True, help="CSV with 128×128 labels")
    ap.add_argument("--tiles_224_dir", required=True, help="Directory with 224×224 tiles")
    ap.add_argument("--out_csv", required=True, help="Output CSV with transferred labels")
    ap.add_argument("--overlap_threshold", type=float, default=0.3, 
                   help="Minimum overlap fraction to transfer label (default: 0.3)")
    args = ap.parse_args()

    print("="*60)
    print("LABEL TRANSFER: 128×128 → 224×224")
    print("="*60)

    # Load 128×128 labels
    df_128 = pd.read_csv(args.labels_128)
    df_128 = df_128[df_128["label_name"].notna()].copy()
    
    print(f"\n128×128 labeled tiles: {len(df_128)}")
    print(f"Label distribution:")
    print(df_128["label_name"].value_counts())

    # Parse 128×128 tile info
    print("\nParsing 128×128 tile metadata...")
    tile_info_128 = []
    for _, row in tqdm(df_128.iterrows(), total=len(df_128)):
        info = parse_tile_name(row["tile_path"])
        if info:
            info['label_name'] = row["label_name"]
            info['label_id'] = row.get("label_id", "")
            tile_info_128.append(info)
    
    df_128_parsed = pd.DataFrame(tile_info_128)
    print(f"Successfully parsed: {len(df_128_parsed)} / {len(df_128)}")

    # Find all 224×224 tiles
    tiles_224_dir = Path(args.tiles_224_dir)
    tile_files_224 = list(tiles_224_dir.glob("*.jpg")) + list(tiles_224_dir.glob("*.png"))
    
    print(f"\n224×224 tiles found: {len(tile_files_224)}")

    # Parse 224×224 tile info
    print("\nParsing 224×224 tile metadata...")
    tile_info_224 = []
    for tile_path in tqdm(tile_files_224):
        info = parse_tile_name(tile_path.name)
        if info:
            info['tile_path'] = str(tile_path)
            tile_info_224.append(info)
    
    df_224 = pd.DataFrame(tile_info_224)
    print(f"Successfully parsed: {len(df_224)} / {len(tile_files_224)}")

    # Transfer labels by matching video_id + frame_idx + spatial overlap
    print(f"\nTransferring labels (overlap threshold: {args.overlap_threshold})...")
    
    labeled_count = 0
    rows_224_labeled = []
    
    for _, tile_224 in tqdm(df_224.iterrows(), total=len(df_224)):
        # Find candidate 128×128 tiles from same video and frame
        candidates = df_128_parsed[
            (df_128_parsed["video_id"] == tile_224["video_id"]) &
            (df_128_parsed["frame_idx"] == tile_224["frame_idx"])
        ]
        
        if len(candidates) == 0:
            # No labeled tiles from this video/frame
            rows_224_labeled.append({
                "tile_path": tile_224["tile_path"],
                "label_id": "",
                "label_name": "",
                "video_id": tile_224["video_id"],
                "frame_idx": tile_224["frame_idx"],
                "x": tile_224["x"],
                "y": tile_224["y"],
                "tile": 224
            })
            continue
        
        # Check spatial overlap with each candidate
        best_overlap = 0
        best_label = None
        
        for _, tile_128 in candidates.iterrows():
            if tiles_overlap(
                tile_224["x"], tile_224["y"], 224,
                tile_128["x"], tile_128["y"], 128,
                threshold=args.overlap_threshold
            ):
                # Use this label (you could also track overlap fraction and pick best)
                best_label = tile_128["label_name"]
                labeled_count += 1
                break
        
        rows_224_labeled.append({
            "tile_path": tile_224["tile_path"],
            "label_id": "" if best_label is None else str(["clean", "unclean", "ignore"].index(best_label)),
            "label_name": "" if best_label is None else best_label,
            "video_id": tile_224["video_id"],
            "frame_idx": tile_224["frame_idx"],
            "x": tile_224["x"],
            "y": tile_224["y"],
            "tile": 224
        })
    
    # Create output dataframe
    df_out = pd.DataFrame(rows_224_labeled)
    
    # Save
    output_path = Path(args.out_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    
    # Summary
    print(f"\n{'='*60}")
    print("TRANSFER COMPLETE")
    print("="*60)
    print(f"Total 224×224 tiles: {len(df_out)}")
    print(f"Labeled tiles:       {labeled_count}")
    print(f"Unlabeled tiles:     {len(df_out) - labeled_count}")
    print(f"\nLabel distribution (transferred):")
    print(df_out[df_out["label_name"] != ""]["label_name"].value_counts())
    print(f"\nOutput saved to: {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Review the transferred labels")
    print(f"  2. Optionally label the remaining unlabeled tiles")
    print(f"  3. Split into train/val/test with split_by_video.py")
    print("="*60)


if __name__ == "__main__":
    main()
