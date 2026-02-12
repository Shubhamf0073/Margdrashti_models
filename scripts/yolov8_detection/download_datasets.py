"""
Download and prepare datasets for YOLOv8 pothole and crack detection.

This script downloads datasets from Roboflow and Kaggle, merges them into
a unified YOLO format structure, and creates the data.yaml configuration file.

Usage:
    python scripts/yolov8_detection/download_datasets.py \
        --roboflow_key YOUR_API_KEY \
        --output_dir data/pothole_crack_detection \
        --split_ratio 0.7 0.2 0.1

Requirements:
    - Roboflow API key (get from https://app.roboflow.com/settings/api)
    - Kaggle API credentials (see https://github.com/Kaggle/kaggle-api#api-credentials)
"""

import argparse
import json
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import yaml

try:
    from roboflow import Roboflow
except ImportError:
    print("WARNING: roboflow not installed. Install with: pip install roboflow")
    Roboflow = None

# NOTE: Kaggle import is deferred to download_kaggle_datasets() because
# the kaggle library triggers authentication at import time, which crashes
# if ~/.kaggle/kaggle.json is missing.
KaggleApi = None


class DatasetDownloader:
    """Download and merge multiple pothole/crack detection datasets."""

    def __init__(self, output_dir, roboflow_key=None):
        self.output_dir = Path(output_dir)
        self.roboflow_key = roboflow_key
        self.temp_dir = self.output_dir / "temp_downloads"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Class mapping for unified dataset
        # Maps various dataset class names to our standard classes
        self.class_mapping = {
            'pothole': 0,
            'Pothole': 0,
            'POTHOLE': 0,
            'pot-hole': 0,
            'crack': 1,
            'Crack': 1,
            'CRACK': 1,
            'longitudinal_crack': 2,
            'Longitudinal Crack': 2,
            'longitudinal-crack': 2,
            'transverse_crack': 3,
            'Transverse Crack': 3,
            'transverse-crack': 3,
            'alligator_crack': 1,  # Map to general crack
            'lateral_crack': 3,  # Map to transverse crack
        }

        self.class_names = ['pothole', 'crack', 'longitudinal_crack', 'transverse_crack']

        print(f"Output directory: {self.output_dir}")
        print(f"Temporary downloads: {self.temp_dir}")

    def download_roboflow_datasets(self):
        """Download multiple datasets from Roboflow."""
        if not Roboflow or not self.roboflow_key:
            print("Skipping Roboflow downloads (no API key or library not installed)")
            return []

        print("\n" + "="*60)
        print("DOWNLOADING ROBOFLOW DATASETS")
        print("="*60)

        rf = Roboflow(api_key=self.roboflow_key)
        downloaded = []

        # List of recommended Roboflow datasets
        datasets = [
            {
                'workspace': 'gerapothole',
                'project': 'pothole-detection-yolov8',
                'version': 1,
                'name': 'GeraPotHole'
            },
            {
                'workspace': 'kartik-zvust',
                'project': 'pothole-detection-yolo-v8',
                'version': 1,
                'name': 'Kartik'
            },
            # Add more datasets as needed
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
                print(f"✓ Downloaded {ds_info['name']} to {dataset.location}")
            except Exception as e:
                print(f"✗ Failed to download {ds_info['name']}: {e}")

        return downloaded

    def download_kaggle_datasets(self):
        """Download datasets from Kaggle."""
        global KaggleApi
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except (ImportError, OSError) as e:
            print(f"Skipping Kaggle downloads (kaggle not available: {e})")
            KaggleApi = None
            return []

        if not KaggleApi:
            print("Skipping Kaggle downloads (kaggle library not installed)")
            return []

        print("\n" + "="*60)
        print("DOWNLOADING KAGGLE DATASETS")
        print("="*60)

        api = KaggleApi()
        api.authenticate()

        downloaded = []

        # List of Kaggle datasets
        kaggle_datasets = [
            {
                'dataset': 'chitholian/annotated-potholes-dataset',
                'name': 'annotated-potholes',
                'has_labels': True
            },
            {
                'dataset': 'sachinpatel21/pothole-image-dataset',
                'name': 'pothole-images',
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

    def convert_kaggle_to_yolo(self, dataset_info):
        """Convert Kaggle dataset annotations to YOLO format."""
        print(f"\nConverting {dataset_info['name']} to YOLO format...")

        # This is a placeholder - actual conversion depends on the annotation format
        # Common formats: COCO JSON, Pascal VOC XML, CSV

        # For annotated-potholes dataset (usually has XML annotations)
        if 'annotated-potholes' in dataset_info['name']:
            print("Note: This dataset may need manual annotation conversion.")
            print("Check the dataset structure and implement conversion if needed.")

        return dataset_info['path']

    def merge_datasets(self, downloaded_datasets, split_ratio=(0.7, 0.2, 0.1)):
        """
        Merge multiple datasets into unified YOLO structure.

        Args:
            downloaded_datasets: List of dataset info dicts
            split_ratio: Tuple of (train, val, test) ratios
        """
        print("\n" + "="*60)
        print("MERGING DATASETS")
        print("="*60)

        # Collect all images and labels
        all_data = []

        for ds in downloaded_datasets:
            print(f"\nProcessing {ds['name']}...")

            if ds['format'] == 'yolov8':
                # Roboflow datasets are already in YOLO format
                ds_path = ds['path']

                for split in ['train', 'valid', 'test']:
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
                                    all_data.append({
                                        'image': img_file,
                                        'label': lbl_file,
                                        'source': ds['name']
                                    })

            elif ds['format'] == 'kaggle':
                # Handle Kaggle datasets (may need conversion)
                print(f"Kaggle dataset {ds['name']} may need manual processing")
                # Add conversion logic here based on specific dataset format

        print(f"\nTotal images collected: {len(all_data)}")

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
                self._normalize_label(item['label'], lbl_dest)

        print("\n✓ Dataset merge complete!")
        print(f"  Train: {len(splits['train'])} images")
        print(f"  Val:   {len(splits['val'])} images")
        print(f"  Test:  {len(splits['test'])} images")

        return splits

    def _normalize_label(self, src_label, dst_label):
        """Normalize label file to use standard class IDs."""
        with open(src_label, 'r') as f_in, open(dst_label, 'w') as f_out:
            for line in f_in:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # YOLO format: class_id x_center y_center width height
                    class_id = int(parts[0])
                    # Map to our standard class IDs if needed
                    # For now, keep the class_id as is
                    # TODO: Implement class mapping based on dataset metadata
                    f_out.write(line)

    def create_data_yaml(self, splits):
        """Create data.yaml configuration for YOLOv8."""
        yaml_path = self.output_dir / 'data.yaml'

        data = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.class_names),
            'names': self.class_names
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

        print(f"\n✓ Created {yaml_path}")
        print("\ndata.yaml contents:")
        print(yaml.dump(data, default_flow_style=False))

        return yaml_path

    def validate_dataset(self):
        """Validate the merged dataset."""
        print("\n" + "="*60)
        print("VALIDATING DATASET")
        print("="*60)

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

            # Sample check label format
            if labels:
                with open(labels[0], 'r') as f:
                    first_line = f.readline().strip()
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
            print(f"\nCleaning up temporary files in {self.temp_dir}...")
            shutil.rmtree(self.temp_dir)
            print("✓ Cleanup complete")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare datasets for YOLOv8 pothole detection"
    )
    parser.add_argument(
        '--roboflow_key',
        type=str,
        help='Roboflow API key (get from https://app.roboflow.com/settings/api)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/pothole_crack_detection',
        help='Output directory for merged dataset (default: data/pothole_crack_detection)'
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

    print("="*60)
    print("YOLOV8 DATASET PREPARATION")
    print("="*60)

    # Initialize downloader
    downloader = DatasetDownloader(
        output_dir=args.output_dir,
        roboflow_key=args.roboflow_key
    )

    # Download datasets
    downloaded = []

    if not args.skip_roboflow:
        roboflow_datasets = downloader.download_roboflow_datasets()
        downloaded.extend(roboflow_datasets)

    if not args.skip_kaggle:
        kaggle_datasets = downloader.download_kaggle_datasets()
        downloaded.extend(kaggle_datasets)

    if not downloaded:
        print("\n⚠ No datasets downloaded!")
        print("\nOptions:")
        print("  1. Provide --roboflow_key to download Roboflow datasets")
        print("  2. Configure Kaggle API credentials (~/.kaggle/kaggle.json)")
        print("  3. Manually download datasets and place them in the temp directory")
        return

    # Merge datasets
    splits = downloader.merge_datasets(downloaded, tuple(args.split_ratio))

    # Create data.yaml
    downloader.create_data_yaml(splits)

    # Validate
    downloader.validate_dataset()

    # Cleanup
    if not args.no_cleanup:
        downloader.cleanup()

    print("\n" + "="*60)
    print("DATASET PREPARATION COMPLETE!")
    print("="*60)
    print(f"\nDataset ready at: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Review the data.yaml file")
    print("  2. Visualize some samples to verify annotations")
    print("  3. Start training with train_yolov8.py")


if __name__ == '__main__':
    main()
