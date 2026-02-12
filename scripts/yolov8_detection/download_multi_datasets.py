"""
Download and merge multiple pothole and crack detection datasets.

This script downloads datasets from Roboflow and Kaggle, then merges them
into a unified YOLO format with all 4 classes (pothole, crack, longitudinal_crack, transverse_crack).

Usage:
    python scripts/yolov8_detection/download_multi_datasets.py \
        --roboflow_key YOUR_API_KEY \
        --output_dir data/pothole_crack_detection_v2 \
        --target_size 4000

Requirements:
    - Roboflow API key: https://app.roboflow.com/settings/api
    - Kaggle API credentials: ~/.kaggle/kaggle.json
"""

import argparse
import json
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict, Counter
import yaml

try:
    from roboflow import Roboflow
except ImportError:
    print("WARNING: roboflow not installed. Install with: pip install roboflow")
    Roboflow = None

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("WARNING: kaggle not installed. Install with: pip install kaggle")
    KaggleApi = None


def print_separator(title=""):
    """Print a formatted separator line."""
    if title:
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print("="*60)
    else:
        print("="*60)


class MultiDatasetMerger:
    """Download and merge multiple datasets for comprehensive training."""

    def __init__(self, output_dir, roboflow_key=None, target_size=4000):
        self.output_dir = Path(output_dir)
        self.roboflow_key = roboflow_key
        self.target_size = target_size
        self.temp_dir = self.output_dir / "temp_downloads"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Standard class mapping (4 classes)
        self.class_mapping = {
            # Pothole variations
            'pothole': 0,
            'Pothole': 0,
            'POTHOLE': 0,
            'pot-hole': 0,
            'potholes': 0,

            # General crack
            'crack': 1,
            'Crack': 1,
            'CRACK': 1,
            'cracks': 1,
            'alligator_crack': 1,
            'alligator': 1,

            # Longitudinal crack
            'longitudinal_crack': 2,
            'Longitudinal Crack': 2,
            'longitudinal-crack': 2,
            'longitudinal': 2,
            'L-crack': 2,

            # Transverse crack
            'transverse_crack': 3,
            'Transverse Crack': 3,
            'transverse-crack': 3,
            'transverse': 3,
            'T-crack': 3,
            'lateral_crack': 3,
            'lateral': 3,
        }

        self.class_names = ['pothole', 'crack', 'longitudinal_crack', 'transverse_crack']

        print(f"Output directory: {self.output_dir}")
        print(f"Target dataset size: {self.target_size} images")
        print(f"Classes: {self.class_names}")

    def download_roboflow_datasets(self):
        """Download multiple datasets from Roboflow."""
        if not Roboflow or not self.roboflow_key:
            print("Skipping Roboflow downloads (no API key or library not installed)")
            return []

        print_separator("DOWNLOADING ROBOFLOW DATASETS")

        rf = Roboflow(api_key=self.roboflow_key)
        downloaded = []

        # Recommended Roboflow datasets
        datasets = [
            {
                'workspace': 'gerapothole',
                'project': 'pothole-detection-yolov8',
                'version': 1,
                'name': 'GeraPotHole_Pothole'
            },
            {
                'workspace': 'kartik-zvust',
                'project': 'pothole-detection-yolo-v8',
                'version': 1,
                'name': 'Kartik_Pothole'
            },
            {
                'workspace': 'intel-unnati-training-program',
                'project': 'pothole-detection-bqu6s',
                'version': 1,
                'name': 'Intel_Pothole'
            },
            # Add road damage datasets with cracks
            # Note: You may need to browse Roboflow Universe to find datasets with crack annotations
        ]

        for ds_info in datasets:
            try:
                print(f"\nDownloading {ds_info['name']}...")
                project = rf.workspace(ds_info['workspace']).project(ds_info['project'])
                dataset = project.version(ds_info['version']).download(
                    "yolov8",
                    location=str(self.temp_dir / ds_info['name'])
                )
                downloaded.append({
                    'name': ds_info['name'],
                    'path': Path(dataset.location),
                    'format': 'yolov8'
                })
                print(f"✓ Downloaded {ds_info['name']}")
            except Exception as e:
                print(f"✗ Failed to download {ds_info['name']}: {e}")

        return downloaded

    def download_kaggle_datasets(self):
        """Download datasets from Kaggle."""
        if not KaggleApi:
            print("Skipping Kaggle downloads (kaggle library not installed)")
            return []

        print_separator("DOWNLOADING KAGGLE DATASETS")

        api = KaggleApi()
        api.authenticate()

        downloaded = []

        kaggle_datasets = [
            {
                'dataset': 'chitholian/annotated-potholes-dataset',
                'name': 'Kaggle_Annotated_Potholes',
                'has_labels': True
            },
            {
                'dataset': 'sachinpatel21/pothole-image-dataset',
                'name': 'Kaggle_Sachin_Potholes',
                'has_labels': True
            },
        ]

        for ds_info in kaggle_datasets:
            try:
                print(f"\nDownloading {ds_info['name']}...")
                download_path = self.temp_dir / ds_info['name']
                download_path.mkdir(exist_ok=True)

                api.dataset_download_files(
                    ds_info['dataset'],
                    path=str(download_path),
                    unzip=True
                )

                downloaded.append({
                    'name': ds_info['name'],
                    'path': download_path,
                    'format': 'kaggle',
                    'has_labels': ds_info['has_labels']
                })
                print(f"✓ Downloaded {ds_info['name']}")
            except Exception as e:
                print(f"✗ Failed to download {ds_info['name']}: {e}")

        return downloaded

    def merge_datasets(self, downloaded_datasets, split_ratio=(0.7, 0.2, 0.1)):
        """
        Merge multiple datasets into unified YOLO structure.

        Args:
            downloaded_datasets: List of dataset info dicts
            split_ratio: Tuple of (train, val, test) ratios
        """
        print_separator("MERGING DATASETS")

        all_data = []
        class_stats = Counter()

        for ds in downloaded_datasets:
            print(f"\nProcessing {ds['name']}...")

            if ds['format'] == 'yolov8':
                ds_path = ds['path']

                # Check common YOLOv8 split names
                for split in ['train', 'valid', 'val', 'test']:
                    img_dir = ds_path / split / 'images'
                    lbl_dir = ds_path / split / 'labels'

                    if not img_dir.exists():
                        img_dir = ds_path / split
                        lbl_dir = ds_path / split

                    if img_dir.exists():
                        for img_file in img_dir.glob('*'):
                            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                                lbl_file = lbl_dir / f"{img_file.stem}.txt"
                                if lbl_file.exists():
                                    # Count classes
                                    with open(lbl_file, 'r') as f:
                                        for line in f:
                                            parts = line.strip().split()
                                            if parts:
                                                cls_id = int(parts[0])
                                                if cls_id < len(self.class_names):
                                                    class_stats[self.class_names[cls_id]] += 1

                                    all_data.append({
                                        'image': img_file,
                                        'label': lbl_file,
                                        'source': ds['name']
                                    })

        print(f"\nTotal images collected: {len(all_data)}")
        print(f"Class distribution:")
        for cls, count in class_stats.most_common():
            print(f"  {cls}: {count} instances")

        if len(all_data) < self.target_size:
            print(f"\n⚠ WARNING: Only {len(all_data)} images collected")
            print(f"   Target was {self.target_size} images")
            print(f"   Consider downloading additional datasets")

        # Shuffle and split
        random.shuffle(all_data)

        train_split = int(len(all_data) * split_ratio[0])
        val_split = int(len(all_data) * split_ratio[1])

        splits = {
            'train': all_data[:train_split],
            'val': all_data[train_split:train_split + val_split],
            'test': all_data[train_split + val_split:]
        }

        # Copy files to output structure
        for split_name, split_data in splits.items():
            print(f"\nCopying {len(split_data)} images to {split_name} split...")

            img_out = self.output_dir / 'images' / split_name
            lbl_out = self.output_dir / 'labels' / split_name
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)

            for i, item in enumerate(split_data):
                # Copy image
                img_dest = img_out / f"{split_name}_{i:05d}{item['image'].suffix}"
                shutil.copy2(item['image'], img_dest)

                # Copy and normalize label
                lbl_dest = lbl_out / f"{split_name}_{i:05d}.txt"
                shutil.copy2(item['label'], lbl_dest)

        print("\n✓ Dataset merge complete!")
        print(f"  Train: {len(splits['train'])} images")
        print(f"  Val:   {len(splits['val'])} images")
        print(f"  Test:  {len(splits['test'])} images")

        return splits, class_stats

    def create_data_yaml(self, class_stats):
        """Create data.yaml configuration for YOLOv8."""
        yaml_path = self.output_dir / 'data.yaml'

        data = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.class_names),
            'names': self.class_names,

            # Add class statistics for reference
            'class_distribution': dict(class_stats)
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

        print(f"\n✓ Created {yaml_path}")
        print("\ndata.yaml contents:")
        print(yaml.dump(data, default_flow_style=False))

        return yaml_path

    def validate_dataset(self):
        """Validate the merged dataset."""
        print_separator("VALIDATING DATASET")

        issues = []

        for split in ['train', 'val', 'test']:
            img_dir = self.output_dir / 'images' / split
            lbl_dir = self.output_dir / 'labels' / split

            if not img_dir.exists():
                issues.append(f"Missing {split} images directory")
                continue

            images = list(img_dir.glob('*'))
            labels = list(lbl_dir.glob('*.txt'))

            print(f"\n{split.upper()} split:")
            print(f"  Images: {len(images)}")
            print(f"  Labels: {len(labels)}")

            # Check for images without labels
            missing_labels = []
            for img in images:
                lbl = lbl_dir / f"{img.stem}.txt"
                if not lbl.exists():
                    missing_labels.append(img.name)

            if missing_labels:
                issues.append(f"{split}: {len(missing_labels)} images without labels")
                print(f"  ⚠ {len(missing_labels)} images without labels")

            # Sample check label format
            if labels:
                with open(labels[0], 'r') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        parts = first_line.split()
                        if len(parts) < 5:
                            issues.append(f"{split}: Invalid label format in {labels[0].name}")
                        else:
                            print(f"  Sample label: {first_line}")

        if issues:
            print("\n⚠ VALIDATION ISSUES:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\n✓ Validation passed!")

        return len(issues) == 0

    def cleanup(self):
        """Remove temporary download directory."""
        if self.temp_dir.exists():
            print(f"\nCleaning up temporary files...")
            shutil.rmtree(self.temp_dir)
            print("✓ Cleanup complete")


def main():
    parser = argparse.ArgumentParser(
        description="Download and merge multiple datasets for YOLOv8 pothole and crack detection"
    )
    parser.add_argument(
        '--roboflow_key',
        type=str,
        help='Roboflow API key (from https://app.roboflow.com/settings/api)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/pothole_crack_detection_v2',
        help='Output directory for merged dataset (default: data/pothole_crack_detection_v2)'
    )
    parser.add_argument(
        '--target_size',
        type=int,
        default=4000,
        help='Target number of images to collect (default: 4000)'
    )
    parser.add_argument(
        '--split_ratio',
        nargs=3,
        type=float,
        default=[0.7, 0.2, 0.1],
        help='Train/val/test split ratios (default: 0.7 0.2 0.1)'
    )
    parser.add_argument(
        '--skip_roboflow',
        action='store_true',
        help='Skip Roboflow dataset downloads'
    )
    parser.add_argument(
        '--skip_kaggle',
        action='store_true',
        help='Skip Kaggle dataset downloads'
    )
    parser.add_argument(
        '--no_cleanup',
        action='store_true',
        help='Keep temporary download files'
    )

    args = parser.parse_args()

    print_separator("MULTI-DATASET MERGER FOR YOLOV8")
    print(f"\nTarget: {args.target_size} images")
    print(f"Output: {args.output_dir}")

    # Initialize merger
    merger = MultiDatasetMerger(
        output_dir=args.output_dir,
        roboflow_key=args.roboflow_key,
        target_size=args.target_size
    )

    # Download datasets
    downloaded = []

    if not args.skip_roboflow:
        roboflow_datasets = merger.download_roboflow_datasets()
        downloaded.extend(roboflow_datasets)

    if not args.skip_kaggle:
        kaggle_datasets = merger.download_kaggle_datasets()
        downloaded.extend(kaggle_datasets)

    if not downloaded:
        print("\n⚠ No datasets downloaded!")
        print("\nOptions:")
        print("  1. Provide --roboflow_key to download Roboflow datasets")
        print("  2. Configure Kaggle API credentials (~/.kaggle/kaggle.json)")
        print("  3. Use existing data in temp directory")
        return

    # Merge datasets
    splits, class_stats = merger.merge_datasets(downloaded, tuple(args.split_ratio))

    # Create data.yaml
    merger.create_data_yaml(class_stats)

    # Validate
    merger.validate_dataset()

    # Cleanup
    if not args.no_cleanup:
        merger.cleanup()

    print_separator("DATASET PREPARATION COMPLETE!")
    print(f"\nMerged dataset ready at: {args.output_dir}")

    total_images = sum(len(v) for v in splits.values())
    if total_images >= args.target_size * 0.9:
        print(f"✓ Achieved target: {total_images} / {args.target_size} images")
    else:
        print(f"⚠ Below target: {total_images} / {args.target_size} images")
        print("  Consider downloading additional datasets")

    print("\nNext steps:")
    print("  1. Review the data.yaml file")
    print("  2. Visualize some samples to verify annotations")
    print("  3. Start training:")
    print(f"     python scripts/yolov8_detection/train_yolov8.py \\")
    print(f"         --data {args.output_dir}/data.yaml \\")
    print(f"         --epochs_stage1 20 \\")
    print(f"         --epochs_stage2 40")


if __name__ == '__main__':
    main()
