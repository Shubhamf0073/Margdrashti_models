"""
Two-stage YOLOv8 training for pothole and crack detection.

This script implements transfer learning with two training stages:
  Stage 1: Freeze backbone, train detection head (15 epochs)
  Stage 2: Full fine-tuning (30 epochs)

Usage:
    # Basic usage (no augmentation)
    python scripts/yolov8_detection/train_yolov8.py \
        --data data/pothole_crack_detection/data.yaml \
        --model yolov8n.pt

    # With recommended augmentation (simple - 5-10% better mAP@50)
    python scripts/yolov8_detection/train_yolov8.py \
        --data data/pothole_crack_detection/data.yaml \
        --model yolov8n.pt \
        --augment

    # With custom augmentation parameters
    python scripts/yolov8_detection/train_yolov8.py \
        --data data/pothole_crack_detection/data.yaml \
        --model yolov8n.pt \
        --mosaic 1.0 \
        --mixup 0.15 \
        --degrees 10 \
        --hsv_v 0.5

    # Full example with all parameters
    python scripts/yolov8_detection/train_yolov8.py \
        --data data/pothole_crack_detection/data.yaml \
        --model yolov8n.pt \
        --epochs_stage1 20 \
        --epochs_stage2 40 \
        --batch 32 \
        --imgsz 640 \
        --device 0 \
        --augment

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


def get_augmentation_params(args):
    """
    Build augmentation parameter dict from command-line args.

    If --augment flag is used, applies recommended presets.
    Individual parameters can override presets.
    """
    # Start with no augmentation (YOLOv8 defaults)
    aug_params = {}

    # If --augment flag is set, use recommended presets
    if args.augment:
        aug_params = {
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 10.0,      # More rotation for road damage
            'translate': 0.1,
            'scale': 0.5,
            'fliplr': 0.5,
            'flipud': 0.0,
            'mosaic': 1.0,
            'mixup': 0.15,        # 15% mixup recommended
            'close_mosaic': 15,
        }
        print("\n✓ Using recommended augmentation preset")

    # Override with any explicitly provided parameters
    if args.hsv_h != 0.015: aug_params['hsv_h'] = args.hsv_h
    if args.hsv_s != 0.7: aug_params['hsv_s'] = args.hsv_s
    if args.hsv_v != 0.4: aug_params['hsv_v'] = args.hsv_v
    if args.degrees != 0.0: aug_params['degrees'] = args.degrees
    if args.translate != 0.1: aug_params['translate'] = args.translate
    if args.scale != 0.5: aug_params['scale'] = args.scale
    if args.shear != 0.0: aug_params['shear'] = args.shear
    if args.perspective != 0.0: aug_params['perspective'] = args.perspective
    if args.fliplr != 0.5: aug_params['fliplr'] = args.fliplr
    if args.flipud != 0.0: aug_params['flipud'] = args.flipud
    if args.mosaic != 1.0: aug_params['mosaic'] = args.mosaic
    if args.mixup != 0.0: aug_params['mixup'] = args.mixup
    if args.close_mosaic != 10: aug_params['close_mosaic'] = args.close_mosaic
    if args.copy_paste != 0.0: aug_params['copy_paste'] = args.copy_paste

    return aug_params


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

    # Get augmentation parameters
    aug_params = get_augmentation_params(args)

    # Print augmentation config
    if aug_params:
        print("\nAugmentation enabled:")
        for key, value in aug_params.items():
            print(f"  {key}: {value}")
    else:
        print("\n⚠️  No augmentation enabled (use --augment for better results)")

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
        **aug_params  # Add augmentation parameters
    )

    # Print results
    print("\n" + "="*60)
    print("STAGE 1 COMPLETE")
    print("="*60)

    # Use the actual save directory from the trainer (Ultralytics may prepend runs/detect/)
    best_model = Path(model.trainer.save_dir) / 'weights' / 'best.pt'
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

    # Get augmentation parameters
    aug_params = get_augmentation_params(args)

    # Print augmentation config
    if aug_params:
        print("\nAugmentation enabled:")
        for key, value in aug_params.items():
            print(f"  {key}: {value}")

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
        **aug_params  # Add augmentation parameters
    )

    # Print results
    print("\n" + "="*60)
    print("STAGE 2 COMPLETE")
    print("="*60)

    # Use the actual save directory from the trainer (Ultralytics may prepend runs/detect/)
    best_model = Path(model.trainer.save_dir) / 'weights' / 'best.pt'
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

    # Augmentation parameters
    augment_group = parser.add_argument_group('Augmentation Parameters')

    # Simple on/off flag
    augment_group.add_argument(
        '--augment',
        action='store_true',
        help='Enable data augmentation (uses recommended presets)'
    )

    # HSV augmentation
    augment_group.add_argument('--hsv_h', type=float, default=0.015,
                              help='HSV Hue augmentation (default: 0.015)')
    augment_group.add_argument('--hsv_s', type=float, default=0.7,
                              help='HSV Saturation augmentation (default: 0.7)')
    augment_group.add_argument('--hsv_v', type=float, default=0.4,
                              help='HSV Value/Brightness augmentation (default: 0.4)')

    # Geometric augmentation
    augment_group.add_argument('--degrees', type=float, default=0.0,
                              help='Rotation degrees (default: 0.0, try 5-10 for variation)')
    augment_group.add_argument('--translate', type=float, default=0.1,
                              help='Translation fraction (default: 0.1)')
    augment_group.add_argument('--scale', type=float, default=0.5,
                              help='Scaling fraction (default: 0.5)')
    augment_group.add_argument('--shear', type=float, default=0.0,
                              help='Shear degrees (default: 0.0)')
    augment_group.add_argument('--perspective', type=float, default=0.0,
                              help='Perspective transform (default: 0.0)')

    # Flip augmentation
    augment_group.add_argument('--fliplr', type=float, default=0.5,
                              help='Horizontal flip probability (default: 0.5)')
    augment_group.add_argument('--flipud', type=float, default=0.0,
                              help='Vertical flip probability (default: 0.0)')

    # Advanced mixing
    augment_group.add_argument('--mosaic', type=float, default=1.0,
                              help='Mosaic augmentation probability (default: 1.0)')
    augment_group.add_argument('--mixup', type=float, default=0.0,
                              help='Mixup augmentation probability (default: 0.0, try 0.1-0.15)')
    augment_group.add_argument('--close_mosaic', type=int, default=10,
                              help='Epochs before end to disable mosaic (default: 10)')

    # Other augmentation
    augment_group.add_argument('--copy_paste', type=float, default=0.0,
                              help='Copy-paste augmentation probability (default: 0.0)')

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
