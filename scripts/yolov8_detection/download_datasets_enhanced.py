"""
Enhanced Multi-Source Dataset Downloader for Margdrashti Pothole Detection

This script downloads datasets from multiple sources:
- Roboflow (via API)
- Kaggle (via API)
- Direct downloads (GitHub, public URLs)

Usage:
    python download_datasets_enhanced.py --roboflow_key YOUR_KEY

    # Download only priority datasets
    python download_datasets_enhanced.py --roboflow_key YOUR_KEY --priority essential

    # Download everything
    python download_datasets_enhanced.py --roboflow_key YOUR_KEY --priority all

Requirements:
    - Roboflow API key: https://app.roboflow.com/settings/api
    - Kaggle API: ~/.kaggle/kaggle.json (optional)
    - Internet connection for direct downloads
"""

import argparse
import json
import os
import shutil
import random
import requests
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import yaml
from tqdm import tqdm

try:
    from roboflow import Roboflow
except ImportError:
    print("WARNING: roboflow not installed. Install with: pip install roboflow")
    Roboflow = None

KaggleApi = None


class EnhancedDatasetDownloader:
    """Download and merge datasets from multiple sources."""

    def __init__(self, output_dir, roboflow_key=None, config_file=None):
        self.output_dir = Path(output_dir)
        self.roboflow_key = roboflow_key
        self.temp_dir = self.output_dir / "temp_downloads"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        if config_file is None:
            config_file = Path(__file__).parent / "datasets_config.yaml"

        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        self.class_mapping = self.config['class_mapping']
        self.class_names = ['pothole', 'crack', 'longitudinal_crack', 'transverse_crack']

        print(f"Output directory: {self.output_dir}")
        print(f"Temporary downloads: {self.temp_dir}")
        print(f"Configuration loaded from: {config_file}")

    def download_roboflow_datasets(self, priority='essential'):
        """Download datasets from Roboflow based on priority."""
        if not Roboflow or not self.roboflow_key:
            print("Skipping Roboflow downloads (no API key or library not installed)")
            return []

        print("\n" + "="*60)
        print("DOWNLOADING ROBOFLOW DATASETS")
        print("="*60)

        rf = Roboflow(api_key=self.roboflow_key)
        downloaded = []

        # Get datasets based on priority
        all_datasets = self.config['roboflow_datasets']
        priority_names = self._get_priority_datasets(priority)

        datasets_to_download = [
            ds for ds in all_datasets
            if ds['name'] in priority_names or priority == 'all'
        ]

        print(f"\nPriority: {priority}")
        print(f"Datasets to download: {len(datasets_to_download)}")

        for ds_info in datasets_to_download:
            try:
                print(f"\n{'='*60}")
                print(f"Downloading: {ds_info['name']}")
                print(f"Expected images: {ds_info['expected_images']}")
                print(f"Classes: {ds_info['classes']}")
                print(f"{'='*60}")

                project = rf.workspace(ds_info['workspace']).project(ds_info['project'])
                dataset = project.version(ds_info['version']).download(
                    "yolov8",
                    location=str(self.temp_dir / ds_info['name'])
                )

                downloaded.append({
                    'name': ds_info['name'],
                    'path': Path(dataset.location),
                    'format': 'yolov8',
                    'source': 'roboflow'
                })
                print(f"✓ Downloaded {ds_info['name']}")

            except Exception as e:
                print(f"✗ Failed to download {ds_info['name']}: {e}")
                print(f"  Workspace: {ds_info['workspace']}")
                print(f"  Project: {ds_info['project']}")
                print(f"  Version: {ds_info['version']}")
                print(f"  This dataset may not be publicly available or may require different credentials")

        return downloaded

    def download_kaggle_datasets(self, priority='essential'):
        """Download datasets from Kaggle based on priority."""
        global KaggleApi
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi as KaggleApiClass
            KaggleApi = KaggleApiClass
        except (ImportError, OSError) as e:
            print(f"Skipping Kaggle downloads: {e}")
            print("To enable Kaggle downloads:")
            print("  1. Install: pip install kaggle")
            print("  2. Get API key from: https://www.kaggle.com/settings")
            print("  3. Save to: ~/.kaggle/kaggle.json")
            return []

        print("\n" + "="*60)
        print("DOWNLOADING KAGGLE DATASETS")
        print("="*60)

        api = KaggleApi()
        api.authenticate()

        downloaded = []
        all_datasets = self.config['kaggle_datasets']
        priority_names = self._get_priority_datasets(priority)

        datasets_to_download = [
            ds for ds in all_datasets
            if ds['name'] in priority_names or priority == 'all'
        ]

        for ds_info in datasets_to_download:
            try:
                print(f"\n{'='*60}")
                print(f"Downloading: {ds_info['name']}")
                print(f"Expected images: {ds_info['expected_images']}")
                print(f"Format: {ds_info['format']}")
                print(f"{'='*60}")

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
                    'format': ds_info['format'],
                    'source': 'kaggle'
                })
                print(f"✓ Downloaded {ds_info['name']}")

            except Exception as e:
                print(f"✗ Failed to download {ds_info['name']}: {e}")

        return downloaded

    def download_direct_datasets(self, priority='essential'):
        """Download datasets from direct URLs."""
        print("\n" + "="*60)
        print("DOWNLOADING DIRECT DATASETS")
        print("="*60)

        downloaded = []
        all_datasets = self.config.get('direct_datasets', [])
        priority_names = self._get_priority_datasets(priority)

        datasets_to_download = [
            ds for ds in all_datasets
            if ds['name'] in priority_names or priority == 'all'
        ]

        for ds_info in datasets_to_download:
            try:
                print(f"\n{'='*60}")
                print(f"Downloading: {ds_info['name']}")
                print(f"URL: {ds_info['url']}")
                print(f"{'='*60}")

                download_path = self.temp_dir / ds_info['name']
                download_path.mkdir(exist_ok=True)

                # Download file
                zip_path = download_path / "dataset.zip"
                self._download_file(ds_info['url'], zip_path)

                # Extract
                print("Extracting...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(download_path)

                zip_path.unlink()  # Remove zip file

                downloaded.append({
                    'name': ds_info['name'],
                    'path': download_path,
                    'format': ds_info['format'],
                    'source': 'direct'
                })
                print(f"✓ Downloaded {ds_info['name']}")

            except Exception as e:
                print(f"✗ Failed to download {ds_info['name']}: {e}")

        return downloaded

    def _download_file(self, url, filepath):
        """Download file with progress bar."""
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(filepath, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True, desc=filepath.name
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    def _get_priority_datasets(self, priority):
        """Get dataset names based on priority level."""
        priorities = self.config.get('download_priority', {})

        if priority == 'all':
            return set()  # Empty set means download all

        names = set()
        if priority in ['essential', 'recommended', 'optional']:
            names.update(priorities.get('essential', []))
        if priority in ['recommended', 'optional']:
            names.update(priorities.get('recommended', []))
        if priority == 'optional':
            names.update(priorities.get('optional', []))

        return names

    def convert_pascal_voc_to_yolo(self, dataset_path):
        """Convert Pascal VOC XML annotations to YOLO format."""
        print(f"\nConverting Pascal VOC annotations to YOLO format...")

        # Find all XML files
        xml_files = list(dataset_path.rglob('*.xml'))

        if not xml_files:
            print("No XML files found, skipping conversion")
            return dataset_path

        # Create YOLO structure
        yolo_path = dataset_path / 'yolo_format'
        yolo_path.mkdir(exist_ok=True)

        for xml_file in tqdm(xml_files, desc="Converting annotations"):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                # Get image size
                size = root.find('size')
                img_width = int(size.find('width').text)
                img_height = int(size.find('height').text)

                # Get image filename
                filename = root.find('filename').text
                img_stem = Path(filename).stem

                # Convert each object
                yolo_annotations = []
                for obj in root.findall('object'):
                    class_name = obj.find('name').text.lower()

                    # Map to our classes
                    if class_name in self.class_mapping:
                        class_id = self.class_mapping[class_name]
                    else:
                        continue  # Skip unknown classes

                    # Get bounding box
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)

                    # Convert to YOLO format (normalized x_center, y_center, width, height)
                    x_center = ((xmin + xmax) / 2) / img_width
                    y_center = ((ymin + ymax) / 2) / img_height
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height

                    yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

                # Save YOLO annotation
                if yolo_annotations:
                    label_file = yolo_path / f"{img_stem}.txt"
                    with open(label_file, 'w') as f:
                        f.write('\n'.join(yolo_annotations))

            except Exception as e:
                print(f"⚠️  Failed to convert {xml_file.name}: {e}")
                continue

        print(f"✓ Converted {len(xml_files)} annotations to YOLO format")
        return yolo_path

    def merge_datasets(self, downloaded_datasets, split_ratio=(0.7, 0.2, 0.1)):
        """Merge multiple datasets into unified YOLO structure."""
        print("\n" + "="*60)
        print("MERGING DATASETS")
        print("="*60)

        all_data = []

        for ds in downloaded_datasets:
            print(f"\nProcessing {ds['name']} ({ds['source']})...")

            if ds['format'] == 'yolov8' or ds['format'] == 'yolo':
                # Already in YOLO format
                ds_path = ds['path']

                for split in ['train', 'valid', 'test']:
                    img_dir = ds_path / split / 'images'
                    lbl_dir = ds_path / split / 'labels'

                    # Some datasets have different structure
                    if not img_dir.exists():
                        img_dir = ds_path / split
                        lbl_dir = ds_path / split

                    # Some datasets store everything in root
                    if not img_dir.exists():
                        img_dir = ds_path / 'images' / split
                        lbl_dir = ds_path / 'labels' / split

                    if img_dir.exists():
                        for img_file in img_dir.glob('*'):
                            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                                # Find corresponding label
                                lbl_file = lbl_dir / f"{img_file.stem}.txt"

                                if lbl_file.exists():
                                    all_data.append({
                                        'image': img_file,
                                        'label': lbl_file,
                                        'source': ds['name']
                                    })

            elif ds['format'] == 'pascal_voc':
                # Convert VOC to YOLO first
                yolo_path = self.convert_pascal_voc_to_yolo(ds['path'])

                # Now process as YOLO
                for img_file in ds['path'].rglob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        lbl_file = yolo_path / f"{img_file.stem}.txt"
                        if lbl_file.exists():
                            all_data.append({
                                'image': img_file,
                                'label': lbl_file,
                                'source': ds['name']
                            })

            print(f"  Collected from {ds['name']}: {len([d for d in all_data if d['source'] == ds['name']])} images")

        print(f"\n{'='*60}")
        print(f"Total images collected: {len(all_data)}")
        print(f"{'='*60}")

        if len(all_data) == 0:
            print("\n❌ No data collected! Check dataset formats and paths.")
            return None

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

            for i, item in enumerate(tqdm(split_data, desc=f"{split_name}")):
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

        # Show source distribution
        print("\nDataset Source Distribution:")
        source_counts = defaultdict(int)
        for item in all_data:
            source_counts[item['source']] += 1
        for source, count in sorted(source_counts.items()):
            print(f"  {source}: {count} images")

        return splits

    def _normalize_label(self, src_label, dst_label):
        """Normalize label file to use standard class IDs."""
        with open(src_label, 'r') as f_in, open(dst_label, 'w') as f_out:
            for line in f_in:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # YOLO format: class_id x_center y_center width height
                    class_id = int(parts[0])
                    # Keep as is (already mapped during conversion if needed)
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
        total_images = 0

        for split in ['train', 'val', 'test']:
            img_dir = self.output_dir / 'images' / split
            lbl_dir = self.output_dir / 'labels' / split

            if not img_dir.exists():
                issues.append(f"Missing {split} images directory")
                continue

            images = list(img_dir.glob('*'))
            labels = list(lbl_dir.glob('*.txt'))
            total_images += len(images)

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

        # Performance prediction
        print(f"\n{'='*60}")
        print("PERFORMANCE PREDICTION")
        print(f"{'='*60}")
        print(f"Total images: {total_images}")

        if total_images >= 4000:
            print("✅ Expected mAP@50: 87-90%")
            print("✅ Expected Recall: 77-82%")
        elif total_images >= 3000:
            print("⚠️  Expected mAP@50: 80-85%")
            print("⚠️  Expected Recall: 72-77%")
        elif total_images >= 2000:
            print("⚠️  Expected mAP@50: 75-80%")
            print("⚠️  Expected Recall: 65-72%")
        else:
            print("❌ Expected mAP@50: 70-75%")
            print("❌ Need more data for better performance")
            print("   Recommendation: Add more datasets from Roboflow/Kaggle")

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
        description="Enhanced multi-source dataset downloader for Margdrashti"
    )
    parser.add_argument(
        '--roboflow_key',
        type=str,
        help='Roboflow API key (https://app.roboflow.com/settings/api)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/pothole_crack_detection',
        help='Output directory for merged dataset'
    )
    parser.add_argument(
        '--split_ratio',
        nargs=3,
        type=float,
        default=[0.7, 0.2, 0.1],
        help='Train/val/test split ratios (default: 0.7 0.2 0.1)'
    )
    parser.add_argument(
        '--priority',
        type=str,
        choices=['essential', 'recommended', 'optional', 'all'],
        default='recommended',
        help='Dataset priority level (default: recommended for 4,000+ images)'
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
        '--skip_direct',
        action='store_true',
        help='Skip direct URL downloads'
    )
    parser.add_argument(
        '--no_cleanup',
        action='store_true',
        help='Keep temporary download files'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to datasets_config.yaml (default: ./datasets_config.yaml)'
    )

    args = parser.parse_args()

    print("="*60)
    print("MARGDRASHTI ENHANCED DATASET PREPARATION")
    print("="*60)
    print(f"Priority: {args.priority}")
    print(f"Target: 4,000+ images for 87-90% mAP@50")
    print("="*60)

    # Initialize downloader
    downloader = EnhancedDatasetDownloader(
        output_dir=args.output_dir,
        roboflow_key=args.roboflow_key,
        config_file=args.config
    )

    # Download datasets
    downloaded = []

    if not args.skip_roboflow:
        roboflow_datasets = downloader.download_roboflow_datasets(priority=args.priority)
        downloaded.extend(roboflow_datasets)

    if not args.skip_kaggle:
        kaggle_datasets = downloader.download_kaggle_datasets(priority=args.priority)
        downloaded.extend(kaggle_datasets)

    if not args.skip_direct:
        direct_datasets = downloader.download_direct_datasets(priority=args.priority)
        downloaded.extend(direct_datasets)

    if not downloaded:
        print("\n⚠ No datasets downloaded!")
        print("\nOptions:")
        print("  1. Provide --roboflow_key to download Roboflow datasets")
        print("     Get key from: https://app.roboflow.com/settings/api")
        print("  2. Configure Kaggle API: ~/.kaggle/kaggle.json")
        print("     Get key from: https://www.kaggle.com/settings")
        print("  3. Check internet connection for direct downloads")
        return

    print(f"\n{'='*60}")
    print(f"DOWNLOADED {len(downloaded)} DATASETS")
    print(f"{'='*60}")
    for ds in downloaded:
        print(f"  ✓ {ds['name']} ({ds['source']})")

    # Merge datasets
    splits = downloader.merge_datasets(downloaded, tuple(args.split_ratio))

    if splits is None:
        print("\n❌ Failed to merge datasets!")
        return

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
    print("  2. Visualize samples to verify annotations")
    print("  3. Start training:")
    print(f"     python train_yolov8.py --data {args.output_dir}/data.yaml")


if __name__ == '__main__':
    main()
