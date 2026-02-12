"""
Two-stage YOLOv8 training for pothole and crack detection.

This script implements transfer learning with two training stages:
  Stage 1: Freeze backbone, train detection head (15 epochs)
  Stage 2: Full fine-tuning (30 epochs)

Usage:
    # Basic usage
    python scripts/yolov8_detection/train_yolov8.py \
        --data data/pothole_crack_detection/data.yaml \
        --model yolov8n.pt

    # Custom parameters
    python scripts/yolov8_detection/train_yolov8.py \
        --data data/pothole_crack_detection/data.yaml \
        --model yolov8n.pt \
        --epochs_stage1 15 \
        --epochs_stage2 30 \
        --batch 32 \
        --imgsz 640 \
        --device 0

    # Resume from stage 1
    python scripts/yolov8_detection/train_yolov8.py \
        --data data/pothole_crack_detection/data.yaml \
        --resume_stage1 scripts/runs/yolov8n_stage1/weights/best.pt \
        --skip_stage1
"""

import argparse
import torch
import yaml
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Install with: pip install ultralytics")
    exit(1)


def print_separator(title=""):
    """Print a formatted separator line."""
    if title:
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print("="*60)
    else:
        print("="*60)


def check_dataset(data_yaml):
    """Validate dataset configuration."""
    print_separator("DATASET VALIDATION")

    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    with open(data_path, 'r') as f:
        data = yaml.safe_load(f)

    print(f"Dataset: {data_path}")
    print(f"Classes ({data['nc']}): {data['names']}")

    # Check if paths exist
    base_path = Path(data['path'])
    for split in ['train', 'val', 'test']:
        if split in data:
            split_path = base_path / data[split]
            if split_path.exists():
                num_images = len(list(split_path.glob('*')))
                print(f"{split.capitalize():5s}: {num_images} images at {split_path}")
            else:
                print(f"{split.capitalize():5s}: ⚠ Path not found: {split_path}")

    return data


def freeze_model_layers(model, freeze_layers=10):
    """
    Freeze the first N layers of the model (backbone).

    Args:
        model: YOLO model instance
        freeze_layers: Number of layers to freeze from the beginning
    """
    print(f"\nFreezing first {freeze_layers} layers...")

    frozen_count = 0
    for i, (name, param) in enumerate(model.model.named_parameters()):
        if i < freeze_layers:
            param.requires_grad = False
            frozen_count += 1

    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.model.parameters())

    print(f"Frozen parameters: {total - trainable:,}")
    print(f"Trainable parameters: {trainable:,}")

    return model


def train_stage1(args):
    """
    Stage 1: Train with frozen backbone.

    Freezes the first 10 layers (backbone feature extraction) and trains
    only the detection head. This allows the model to adapt to pothole/crack
    detection without forgetting COCO pretrained features.
    """
    print_separator("STAGE 1: FROZEN BACKBONE TRAINING")

    print(f"\nInitializing YOLOv8 from {args.model}...")
    model = YOLO(args.model)

    # Freeze backbone layers
    if args.freeze_layers > 0:
        model = freeze_model_layers(model, args.freeze_layers)

    # Training configuration
    print("\nTraining configuration:")
    print(f"  Epochs: {args.epochs_stage1}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Initial LR: {args.lr0_stage1}")
    print(f"  Device: {args.device}")

    # Start training
    print("\nStarting Stage 1 training...")
    results = model.train(
        data=args.data,
        epochs=args.epochs_stage1,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0_stage1,
        lrf=0.01,  # Final LR = 1% of initial
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        optimizer='AdamW',
        cos_lr=False,  # Linear LR decay for stage 1
        project=args.project,
        name='yolov8n_stage1',
        exist_ok=True,
        pretrained=True,
        device=args.device,
        workers=args.workers,
        patience=args.patience_stage1,
        save=True,
        save_period=5,
        cache=args.cache,
        val=True,
        plots=True,
        verbose=True,
        seed=0,
        deterministic=False,
        single_cls=False,
        rect=False,
        resume=False,
        amp=True,  # Automatic Mixed Precision
    )

    # Print results
    print("\n" + "="*60)
    print("STAGE 1 COMPLETE")
    print("="*60)

    best_model = Path(args.project) / 'yolov8n_stage1' / 'weights' / 'best.pt'
    print(f"\nBest model saved to: {best_model}")

    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"\nFinal metrics:")
        print(f"  mAP@50: {metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP@50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"  Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
        print(f"  Recall: {metrics.get('metrics/recall(B)', 0):.4f}")

    return best_model


def train_stage2(args, stage1_model):
    """
    Stage 2: Full fine-tuning.

    Unfreezes all layers and fine-tunes the entire model with a lower
    learning rate. Uses cosine annealing LR schedule for smooth convergence.
    """
    print_separator("STAGE 2: FULL FINE-TUNING")

    print(f"\nLoading model from Stage 1: {stage1_model}...")
    model = YOLO(str(stage1_model))

    # Unfreeze all layers
    print("\nUnfreezing all layers...")
    for param in model.model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    # Training configuration
    print("\nTraining configuration:")
    print(f"  Epochs: {args.epochs_stage2}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Initial LR: {args.lr0_stage2} (10x lower than Stage 1)")
    print(f"  LR schedule: Cosine annealing")
    print(f"  Device: {args.device}")

    # Start training
    print("\nStarting Stage 2 training...")
    results = model.train(
        data=args.data,
        epochs=args.epochs_stage2,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0_stage2,
        lrf=0.0001,  # Very low final LR
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=0,  # No warmup for stage 2
        optimizer='AdamW',
        cos_lr=True,  # Cosine annealing LR schedule
        project=args.project,
        name='yolov8n_stage2',
        exist_ok=True,
        pretrained=False,  # Already pretrained from stage 1
        device=args.device,
        workers=args.workers,
        patience=args.patience_stage2,
        save=True,
        save_period=5,
        cache=args.cache,
        val=True,
        plots=True,
        verbose=True,
        seed=0,
        deterministic=False,
        single_cls=False,
        rect=False,
        resume=False,
        amp=True,
    )

    # Print results
    print("\n" + "="*60)
    print("STAGE 2 COMPLETE")
    print("="*60)

    best_model = Path(args.project) / 'yolov8n_stage2' / 'weights' / 'best.pt'
    print(f"\nBest model saved to: {best_model}")

    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"\nFinal metrics:")
        print(f"  mAP@50: {metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP@50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"  Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
        print(f"  Recall: {metrics.get('metrics/recall(B)', 0):.4f}")

    return best_model


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage YOLOv8 training for pothole and crack detection"
    )

    # Dataset
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to data.yaml configuration file'
    )

    # Model
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help='Initial model (default: yolov8n.pt for pretrained COCO weights)'
    )

    # Stage 1 parameters
    parser.add_argument(
        '--epochs_stage1',
        type=int,
        default=15,
        help='Number of epochs for stage 1 (default: 15)'
    )
    parser.add_argument(
        '--lr0_stage1',
        type=float,
        default=0.01,
        help='Initial learning rate for stage 1 (default: 0.01)'
    )
    parser.add_argument(
        '--patience_stage1',
        type=int,
        default=5,
        help='Early stopping patience for stage 1 (default: 5)'
    )
    parser.add_argument(
        '--freeze_layers',
        type=int,
        default=10,
        help='Number of layers to freeze in stage 1 (default: 10)'
    )
    parser.add_argument(
        '--skip_stage1',
        action='store_true',
        help='Skip stage 1 and go directly to stage 2'
    )
    parser.add_argument(
        '--resume_stage1',
        type=str,
        help='Resume from stage 1 checkpoint (use with --skip_stage1)'
    )

    # Stage 2 parameters
    parser.add_argument(
        '--epochs_stage2',
        type=int,
        default=30,
        help='Number of epochs for stage 2 (default: 30)'
    )
    parser.add_argument(
        '--lr0_stage2',
        type=float,
        default=0.001,
        help='Initial learning rate for stage 2 (default: 0.001, 10x lower)'
    )
    parser.add_argument(
        '--patience_stage2',
        type=int,
        default=10,
        help='Early stopping patience for stage 2 (default: 10)'
    )
    parser.add_argument(
        '--skip_stage2',
        action='store_true',
        help='Skip stage 2 (train only stage 1)'
    )

    # Training hyperparameters
    parser.add_argument(
        '--batch',
        type=int,
        default=32,
        help='Batch size (default: 32, adjust based on GPU memory)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size (default: 640)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='Device for training: 0, 1, 2, etc. or cpu (default: 0)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of dataloader workers (default: 8)'
    )
    parser.add_argument(
        '--cache',
        action='store_true',
        help='Cache images in RAM for faster training'
    )

    # Output
    parser.add_argument(
        '--project',
        type=str,
        default='scripts/runs',
        help='Project directory for saving results (default: scripts/runs)'
    )

    args = parser.parse_args()

    # Print configuration
    print_separator("YOLOV8 TWO-STAGE TRAINING")
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Data: {args.data}")
    print(f"  Model: {args.model}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Device: {args.device}")
    print(f"  Output: {args.project}")

    # Check CUDA availability
    if args.device != 'cpu':
        if not torch.cuda.is_available():
            print("\n⚠ WARNING: CUDA not available, falling back to CPU")
            print("Training will be very slow. Consider using Google Colab or a GPU.")
            args.device = 'cpu'
        else:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"\n✓ GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")

    # Validate dataset
    check_dataset(args.data)

    # Stage 1
    if not args.skip_stage1:
        stage1_model = train_stage1(args)
    else:
        if args.resume_stage1:
            stage1_model = Path(args.resume_stage1)
            print(f"\nSkipping Stage 1, using checkpoint: {stage1_model}")
        else:
            print("\nERROR: --skip_stage1 requires --resume_stage1 checkpoint")
            return

    # Stage 2
    if not args.skip_stage2:
        final_model = train_stage2(args, stage1_model)
    else:
        final_model = stage1_model
        print("\nSkipping Stage 2")

    # Final summary
    print_separator("TRAINING COMPLETE")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nFinal model: {final_model}")
    print("\nNext steps:")
    print("  1. Validate the model:")
    print(f"     python scripts/yolov8_detection/validate_yolov8.py --model {final_model} --data {args.data}")
    print("\n  2. Run inference:")
    print(f"     python deploy_inference_yolov8.py --model {final_model} --roi roi_highway_shorter.json --source <video>")
    print("\n  3. Export for deployment:")
    print(f"     python scripts/yolov8_detection/export_model.py --model {final_model}")


if __name__ == '__main__':
    main()
