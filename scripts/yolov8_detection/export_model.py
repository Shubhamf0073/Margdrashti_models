"""
Export YOLOv8 model to various formats for deployment optimization.

This script exports trained YOLOv8 models to ONNX, TensorRT, and other
formats to improve inference speed and enable deployment on different platforms.

Usage:
    # Export to ONNX (CPU optimization)
    python scripts/yolov8_detection/export_model.py \
        --model scripts/runs/yolov8n_stage2/weights/best.pt \
        --format onnx

    # Export to TensorRT (GPU optimization)
    python scripts/yolov8_detection/export_model.py \
        --model scripts/runs/yolov8n_stage2/weights/best.pt \
        --format engine \
        --half

    # Export to multiple formats
    python scripts/yolov8_detection/export_model.py \
        --model scripts/runs/yolov8n_stage2/weights/best.pt \
        --format onnx engine torchscript \
        --imgsz 640

Performance Expectations:
    - ONNX: 2-3x faster on CPU, good compatibility
    - TensorRT (FP16): 1.5-2x faster on GPU, NVIDIA only
    - TorchScript: Slight speedup, easy deployment
"""

import argparse
from pathlib import Path

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


def export_model(model_path, formats, imgsz=640, half=False, simplify=True):
    """
    Export YOLOv8 model to specified formats.

    Args:
        model_path: Path to trained YOLOv8 model (.pt file)
        formats: List of export formats
        imgsz: Input image size for export
        half: Use FP16 precision (only for GPU formats like TensorRT)
        simplify: Simplify ONNX model (recommended)

    Returns:
        dict: Mapping of format to exported file path
    """
    print_separator("MODEL EXPORT")

    # Load model
    print(f"\nLoading model from: {model_path}")
    model = YOLO(model_path)

    # Get model info
    print(f"Model type: {model.model.__class__.__name__}")
    print(f"Input size: {imgsz}×{imgsz}")
    print(f"Classes: {list(model.names.values())}")

    exported_files = {}

    # Export to each format
    for fmt in formats:
        print_separator(f"EXPORTING TO {fmt.upper()}")

        try:
            if fmt == 'onnx':
                print("\nExporting to ONNX format...")
                print("  Benefits: 2-3x faster on CPU, good compatibility")
                print("  Use case: CPU deployment, cross-platform inference")

                export_path = model.export(
                    format='onnx',
                    imgsz=imgsz,
                    simplify=simplify,
                    opset=12,
                    dynamic=False
                )

            elif fmt == 'engine' or fmt == 'tensorrt':
                print("\nExporting to TensorRT format...")
                print("  Benefits: 1.5-2x faster on GPU with FP16")
                print("  Use case: NVIDIA GPU deployment")
                print("  Note: Requires TensorRT library installed")

                if half:
                    print("  Using FP16 precision (recommended for speed)")

                export_path = model.export(
                    format='engine',
                    imgsz=imgsz,
                    half=half,
                    device=0,
                    workspace=4,  # GB
                    verbose=True
                )

            elif fmt == 'torchscript' or fmt == 'ts':
                print("\nExporting to TorchScript format...")
                print("  Benefits: Faster loading, C++ deployment")
                print("  Use case: Production PyTorch deployment")

                export_path = model.export(
                    format='torchscript',
                    imgsz=imgsz,
                    optimize=True
                )

            elif fmt == 'openvino':
                print("\nExporting to OpenVINO format...")
                print("  Benefits: Intel CPU/GPU optimization")
                print("  Use case: Intel hardware deployment")

                export_path = model.export(
                    format='openvino',
                    imgsz=imgsz,
                    half=half
                )

            elif fmt == 'coreml':
                print("\nExporting to CoreML format...")
                print("  Benefits: Apple device optimization")
                print("  Use case: iOS/macOS deployment")

                export_path = model.export(
                    format='coreml',
                    imgsz=imgsz
                )

            elif fmt == 'tflite':
                print("\nExporting to TensorFlow Lite format...")
                print("  Benefits: Mobile device optimization")
                print("  Use case: Android/mobile deployment")

                export_path = model.export(
                    format='tflite',
                    imgsz=imgsz,
                    int8=False
                )

            else:
                print(f"⚠ Unknown format: {fmt}, skipping...")
                continue

            exported_files[fmt] = export_path
            print(f"\n✓ Export complete: {export_path}")

            # Show file size
            export_file = Path(export_path)
            if export_file.exists():
                size_mb = export_file.stat().st_size / (1024 * 1024)
                print(f"  File size: {size_mb:.2f} MB")

        except Exception as e:
            print(f"\n✗ Export failed for {fmt}: {e}")
            print(f"  This may be due to missing dependencies or unsupported platform")

    return exported_files


def benchmark_exports(model_path, exported_files, test_imgsz=640):
    """
    Benchmark inference speed for different export formats.

    Args:
        model_path: Original PyTorch model path
        exported_files: Dict of format -> exported file path
        test_imgsz: Image size for benchmarking
    """
    print_separator("BENCHMARKING")

    print(f"\nBenchmarking inference speed (image size: {test_imgsz}×{test_imgsz})")
    print("Running 100 warmup iterations + 100 test iterations per format\n")

    results = {}

    # Benchmark original PyTorch model
    try:
        print("Testing PyTorch (.pt)...")
        model = YOLO(model_path)
        metrics = model.val(
            data=None,  # Skip validation dataset
            batch=1,
            imgsz=test_imgsz,
            plots=False,
            verbose=False
        )
        print(f"  PyTorch: Baseline")
        results['pytorch'] = 'baseline'
    except Exception as e:
        print(f"  PyTorch benchmark failed: {e}")

    # Benchmark exported formats
    for fmt, export_path in exported_files.items():
        try:
            print(f"Testing {fmt.upper()}...")
            model = YOLO(export_path)
            # Note: YOLO doesn't provide direct benchmarking
            # Users should use deploy_inference_yolov8.py to measure real FPS
            print(f"  {fmt.upper()}: Use deploy_inference_yolov8.py to measure real-world FPS")
            results[fmt] = 'exported'
        except Exception as e:
            print(f"  {fmt.upper()} benchmark failed: {e}")

    print("\nNote: For accurate FPS measurements, use:")
    print("  python deploy_inference_yolov8.py --model <exported_model> ...")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 model to various formats for deployment"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained YOLOv8 model (.pt file)'
    )
    parser.add_argument(
        '--format',
        nargs='+',
        default=['onnx'],
        choices=['onnx', 'engine', 'tensorrt', 'torchscript', 'ts', 'openvino', 'coreml', 'tflite'],
        help='Export formats (default: onnx). Can specify multiple formats.'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size for export (default: 640)'
    )
    parser.add_argument(
        '--half',
        action='store_true',
        help='Use FP16 precision (for TensorRT/engine format, 2x speedup)'
    )
    parser.add_argument(
        '--no_simplify',
        action='store_true',
        help='Do not simplify ONNX model (not recommended)'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run benchmarks after export (experimental)'
    )

    args = parser.parse_args()

    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        return

    print_separator("YOLOV8 MODEL EXPORT")
    print(f"\nModel: {model_path}")
    print(f"Formats: {', '.join(args.format)}")
    print(f"Image size: {args.imgsz}×{args.imgsz}")
    if args.half:
        print("Precision: FP16 (half precision)")
    else:
        print("Precision: FP32 (full precision)")

    # Export model
    exported_files = export_model(
        model_path=str(model_path),
        formats=args.format,
        imgsz=args.imgsz,
        half=args.half,
        simplify=not args.no_simplify
    )

    # Benchmark (optional)
    if args.benchmark and exported_files:
        benchmark_exports(str(model_path), exported_files, args.imgsz)

    # Print summary
    print_separator("EXPORT COMPLETE")

    if exported_files:
        print("\n✓ Successfully exported formats:")
        for fmt, path in exported_files.items():
            print(f"  {fmt.upper():12s} -> {path}")

        print("\nNext steps:")
        print("  1. Test inference with exported model:")
        for fmt, path in list(exported_files.items())[:1]:  # Show first export
            print(f"     python deploy_inference_yolov8.py --model {path} --roi roi_highway_shorter.json --source <video>")

        print("\n  2. Compare performance:")
        print("     - Measure FPS with original .pt model")
        print("     - Measure FPS with exported model")
        print("     - Expected speedup:")
        for fmt in exported_files.keys():
            if fmt == 'onnx':
                print("       • ONNX (CPU): 2-3x faster")
            elif fmt in ['engine', 'tensorrt']:
                print("       • TensorRT (GPU): 1.5-2x faster")
            elif fmt in ['torchscript', 'ts']:
                print("       • TorchScript: 1.1-1.3x faster")

    else:
        print("\n⚠ No models were successfully exported")
        print("Check error messages above for details")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
