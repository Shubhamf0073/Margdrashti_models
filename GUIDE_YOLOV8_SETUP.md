# Complete Guide: Running YOLOv8 Pothole Detection from Start

This guide will walk you through the entire process of setting up and running the YOLOv8 pothole detection system.

## Prerequisites

Ensure you have:
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- Git repository pulled with latest changes

## 🚀 Step-by-Step Guide

### Step 1: Install Dependencies

```bash
# Install required packages
pip install ultralytics roboflow kaggle opencv-python torch torchvision

# Verify installation
python -c "from ultralytics import YOLO; print('✓ Ultralytics installed')"
python -c "import cv2; print('✓ OpenCV installed')"
python -c "import torch; print('✓ PyTorch installed')"
```

### Step 2: Create ROI Configuration

You have two options:

#### Option A: Use Interactive ROI Editor (Recommended)

```bash
# Run the ROI editor with your video
python draw_pothole_roi.py --video path/to/your/video.mp4

# Instructions:
# 1. Click 4-8 points on the video to define the road area polygon
# 2. Points should form a polygon covering only the road surface
# 3. Press 's' to save the ROI to a JSON file
# 4. The file will be saved as roi_config.json
```

#### Option B: Use Pre-configured ROI

If you already have a video with specific dimensions, use this sample ROI configuration:

**For 1920×1080 video (Full HD):**
```json
{
  "roi_points": [
    [300, 700],
    [1620, 700],
    [1400, 400],
    [520, 400]
  ]
}
```

**For 1280×720 video (HD):**
```json
{
  "roi_points": [
    [200, 500],
    [1080, 500],
    [950, 280],
    [330, 280]
  ]
}
```

Save this as `roi_pothole_detection.json` in your project root.

### Step 3: Prepare Dataset

You have three options:

#### Option 3A: Download from Roboflow (Fastest - Recommended)

```bash
# Get your API key from https://app.roboflow.com/settings/api
export ROBOFLOW_API_KEY="your_key_here"

python scripts/yolov8_detection/download_datasets.py \
    --roboflow_key $ROBOFLOW_API_KEY \
    --output_dir data/pothole_crack_detection \
    --split_ratio 0.7 0.2 0.1
```

#### Option 3B: Download from Kaggle

```bash
# First, configure Kaggle API
# Download kaggle.json from https://www.kaggle.com/settings
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download datasets
python scripts/yolov8_detection/download_datasets.py \
    --skip_roboflow \
    --output_dir data/pothole_crack_detection
```

#### Option 3C: Use Sample Dataset for Testing

If you just want to test the system quickly, download a small pre-formatted dataset:

```bash
# Download a sample YOLOv8 pothole dataset
cd data/pothole_crack_detection
wget https://universe.roboflow.com/intel-unnati-training-program/pothole-detection-bqu6s/dataset/1/download -O dataset.zip
unzip dataset.zip
cd ../..

# Verify the structure
ls -la data/pothole_crack_detection/
# Should see: images/, labels/, data.yaml
```

### Step 4: Verify Dataset

```bash
# Check dataset structure
python -c "
import yaml
with open('data/pothole_crack_detection/data.yaml', 'r') as f:
    data = yaml.safe_load(f)
    print('✓ Dataset configuration loaded')
    print(f'  Classes: {data[\"names\"]}')
    print(f'  Number of classes: {data[\"nc\"]}')
"

# Count images
echo "Train images: $(ls data/pothole_crack_detection/images/train/ | wc -l)"
echo "Val images: $(ls data/pothole_crack_detection/images/val/ | wc -l)"
echo "Test images: $(ls data/pothole_crack_detection/images/test/ | wc -l)"
```

Expected output:
```
✓ Dataset configuration loaded
  Classes: ['pothole', 'crack', 'longitudinal_crack', 'transverse_crack']
  Number of classes: 4
Train images: 3500
Val images: 700
Test images: 700
```

### Step 5: Train the Model

#### Quick Training (Testing - 5 epochs)

For testing the pipeline:

```bash
python scripts/yolov8_detection/train_yolov8.py \
    --data data/pothole_crack_detection/data.yaml \
    --model yolov8n.pt \
    --epochs_stage1 3 \
    --epochs_stage2 5 \
    --batch 16 \
    --device 0
```

Training time: ~1 hour

#### Full Training (Production - Recommended)

For best accuracy:

```bash
python scripts/yolov8_detection/train_yolov8.py \
    --data data/pothole_crack_detection/data.yaml \
    --model yolov8n.pt \
    --epochs_stage1 15 \
    --epochs_stage2 30 \
    --batch 32 \
    --device 0 \
    --cache
```

Training time: ~8-10 hours on RTX 3050 Ti

#### Training Options

- `--batch 32`: Adjust based on GPU memory (16 for 6GB GPU, 8 for 4GB)
- `--device 0`: Use GPU 0 (use `--device cpu` for CPU training)
- `--cache`: Cache images in RAM for faster training (requires 16GB+ RAM)
- `--imgsz 640`: Image size (can reduce to 512 for faster training)

#### Monitor Training

Watch the training progress:

```bash
# Training logs are saved to:
tail -f scripts/runs/yolov8n_stage1/train.log
tail -f scripts/runs/yolov8n_stage2/train.log

# View training plots (after training completes)
ls scripts/runs/yolov8n_stage2/
# Look for: results.png, confusion_matrix.png, val_batch0_pred.jpg
```

### Step 6: Validate the Trained Model

After training completes, validate the model:

```bash
# Get the best model path
BEST_MODEL="scripts/runs/yolov8n_stage2/weights/best.pt"

# Run validation
python -c "
from ultralytics import YOLO
model = YOLO('$BEST_MODEL')
results = model.val(data='data/pothole_crack_detection/data.yaml')
print(f'\n✓ Validation complete!')
print(f'  mAP@50: {results.box.map50:.4f}')
print(f'  mAP@50-95: {results.box.map:.4f}')
print(f'  Precision: {results.box.mp:.4f}')
print(f'  Recall: {results.box.mr:.4f}')
"
```

Expected results:
```
✓ Validation complete!
  mAP@50: 0.9100
  mAP@50-95: 0.6800
  Precision: 0.8700
  Recall: 0.8400
```

### Step 7: Run Real-Time Inference

#### Test on Video

```bash
python deploy_inference_yolov8.py \
    --model scripts/runs/yolov8n_stage2/weights/best.pt \
    --roi roi_pothole_detection.json \
    --source path/to/your/video.mp4 \
    --confidence 0.5 \
    --save_output results/output_yolov8.mp4
```

#### Test on Webcam

```bash
python deploy_inference_yolov8.py \
    --model scripts/runs/yolov8n_stage2/weights/best.pt \
    --roi roi_pothole_detection.json \
    --source 0 \
    --confidence 0.5
```

#### Inference Options

- `--confidence 0.5`: Detection confidence threshold (0.3-0.8)
  - Lower = more detections (more false positives)
  - Higher = fewer detections (fewer false positives)
- `--process_every 2`: Process every 2nd frame (2x faster, use for slower GPUs)
- `--no_display`: Don't show video window (for headless servers)
- `--device cpu`: Use CPU instead of GPU

#### Keyboard Controls During Inference

- **'q'**: Quit inference
- **'s'**: Save screenshot
- **'p'**: Pause (press any key to continue)

### Step 8: Compare with Old Model

Test the difference between old EfficientNet and new YOLOv8:

```bash
# Test old EfficientNet model
python deploy_inference_realtime_fast.py \
    --model scripts/runs/efficientnet_224/best.pt \
    --roi roi_pothole_detection.json \
    --source path/to/test_video.mp4 \
    --save_output results/output_efficientnet.mp4

# Test new YOLOv8 model
python deploy_inference_yolov8.py \
    --model scripts/runs/yolov8n_stage2/weights/best.pt \
    --roi roi_pothole_detection.json \
    --source path/to/test_video.mp4 \
    --save_output results/output_yolov8.mp4

# Compare the outputs side-by-side
```

Expected improvements:
- **FPS**: 15-25 → 80-120 (3-5x faster)
- **Accuracy**: ~60% → 90-93% mAP@50
- **False positives**: Significantly reduced
- **Localization**: Precise bounding boxes vs. tile regions

### Step 9: (Optional) Optimize for Deployment

For production deployment, export to optimized formats:

```bash
# Export to ONNX (2-3x faster on CPU)
python scripts/yolov8_detection/export_model.py \
    --model scripts/runs/yolov8n_stage2/weights/best.pt \
    --format onnx

# Export to TensorRT (1.5-2x faster on NVIDIA GPU)
python scripts/yolov8_detection/export_model.py \
    --model scripts/runs/yolov8n_stage2/weights/best.pt \
    --format engine \
    --half

# Use the exported model
python deploy_inference_yolov8.py \
    --model scripts/runs/yolov8n_stage2/weights/best.onnx \
    --roi roi_pothole_detection.json \
    --source video.mp4 \
    --device cpu  # ONNX works great on CPU
```

## 🎯 Quick Start (Minimum Steps)

If you want to test the system quickly with a pre-trained model:

```bash
# 1. Download a pre-trained YOLOv8n model (general purpose)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# 2. Create a simple ROI
cat > roi_test.json << 'EOF'
{
  "roi_points": [[300, 700], [1620, 700], [1400, 400], [520, 400]]
}
EOF

# 3. Test inference (will work but not optimized for potholes)
python deploy_inference_yolov8.py \
    --model yolov8n.pt \
    --roi roi_test.json \
    --source 0 \
    --confidence 0.3
```

## 📊 Troubleshooting

### Issue: CUDA Out of Memory

**Solution 1**: Reduce batch size
```bash
python scripts/yolov8_detection/train_yolov8.py \
    --batch 8  # Reduced from 32
```

**Solution 2**: Reduce image size
```bash
python scripts/yolov8_detection/train_yolov8.py \
    --imgsz 512  # Reduced from 640
```

### Issue: Slow Inference (<30 FPS)

**Solution 1**: Frame skipping
```bash
python deploy_inference_yolov8.py \
    --process_every 2  # Process every 2nd frame
```

**Solution 2**: Export to ONNX/TensorRT
```bash
python scripts/yolov8_detection/export_model.py --model best.pt --format onnx
```

**Solution 3**: Lower resolution
```bash
# Edit deploy_inference_yolov8.py line with predict():
# Change imgsz=640 to imgsz=416
```

### Issue: Too Many False Positives

**Solution**: Increase confidence threshold
```bash
python deploy_inference_yolov8.py \
    --confidence 0.7  # Increased from 0.5
```

### Issue: Missing Detections

**Solution**: Lower confidence threshold
```bash
python deploy_inference_yolov8.py \
    --confidence 0.3  # Decreased from 0.5
```

## 📁 Expected Directory Structure After Setup

```
Margdrashti_models/
├── data/
│   └── pothole_crack_detection/
│       ├── images/
│       │   ├── train/  (3500+ images)
│       │   ├── val/    (700+ images)
│       │   └── test/   (700+ images)
│       ├── labels/
│       │   ├── train/  (3500+ .txt files)
│       │   ├── val/    (700+ .txt files)
│       │   └── test/   (700+ .txt files)
│       └── data.yaml
├── scripts/
│   ├── runs/
│   │   ├── yolov8n_stage1/
│   │   │   └── weights/
│   │   │       └── best.pt
│   │   └── yolov8n_stage2/
│   │       └── weights/
│   │           ├── best.pt      (Use this for inference!)
│   │           ├── best.onnx    (After export)
│   │           └── best.engine  (After TensorRT export)
│   └── yolov8_detection/
│       ├── train_yolov8.py
│       ├── download_datasets.py
│       ├── export_model.py
│       └── README.md
├── deploy_inference_yolov8.py
├── roi_pothole_detection.json
└── results/
    ├── output_yolov8.mp4
    └── output_efficientnet.mp4
```

## 🎓 Training on Google Colab (Faster Alternative)

If local training is too slow, use Google Colab:

1. Upload dataset to Google Drive
2. Create a new Colab notebook
3. Run:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install ultralytics

# Clone your repository
!git clone https://github.com/Shubhamf0073/Margdrashti_models.git
%cd Margdrashti_models

# Copy dataset from Drive
!cp -r /content/drive/MyDrive/pothole_dataset data/pothole_crack_detection

# Train
!python scripts/yolov8_detection/train_yolov8.py \
    --data data/pothole_crack_detection/data.yaml \
    --model yolov8n.pt \
    --epochs_stage1 15 \
    --epochs_stage2 30 \
    --batch 64 \
    --device 0

# Download trained model
from google.colab import files
files.download('scripts/runs/yolov8n_stage2/weights/best.pt')
```

## 🔧 Troubleshooting Low Training Metrics

### Problem: mAP@50 < 80% or Recall < 70%

**Symptoms:**
- Model trained but mAP@50 is 70-75% (target is 90%+)
- Recall is 60-70% (missing 30-40% of potholes!)
- Training plateaus around epoch 20-25
- Only detecting 1 class (pothole) instead of 4

**Root Causes:**

#### 1. Dataset Too Small ⚠️ (Most Common)
```
Your dataset: 2,000-3,000 images → mAP@50: 72-75%
Target dataset: 4,000-5,000 images → mAP@50: 87-90%
```

**Fix:**
```bash
# Download and merge multiple datasets
python scripts/yolov8_detection/download_multi_datasets.py \
    --roboflow_key YOUR_API_KEY \
    --target_size 4000 \
    --output_dir data/pothole_crack_detection_v2

# Retrain with larger dataset
python scripts/yolov8_detection/train_yolov8.py \
    --data data/pothole_crack_detection_v2/data.yaml \
    --epochs_stage1 20 \
    --epochs_stage2 40
```

Expected improvement: 72% → 87-90% mAP@50

#### 2. Missing Classes (Crack Detection)

**Check your data.yaml:**
```bash
cat data/pothole_crack_detection/data.yaml
```

**If you see only 1 class:**
```yaml
nc: 1
names: ['pothole']
```

**You need datasets with all 4 classes:**
- pothole
- crack
- longitudinal_crack
- transverse_crack

**Fix:** Download datasets that include crack annotations (use `download_multi_datasets.py`)

#### 3. Training Plateau (Not Improving After Epoch 20)

**Evidence:**
```
Epoch 20: mAP@50 = 70.5%
Epoch 25: mAP@50 = 71.8%
Epoch 30: mAP@50 = 72.1%  ← Only 1.6% improvement in 10 epochs!
```

**Diagnosis:** Model has learned all it can from limited dataset

**Fix:**
1. **Primary solution:** Add more training data (see Fix #1)
2. **If already have 4,000+ images:** Try larger model:
   ```bash
   --model yolov8s.pt  # instead of yolov8n.pt
   ```

#### 4. Low Recall (Missing Too Many Detections)

**Symptoms:**
- Recall < 70% (missing 30%+ of potholes)
- Precision is OK (75-80%) but recall is low

**Diagnosis:** Model is too conservative (high confidence threshold)

**Fix - Option A: Lower confidence threshold during validation:**
```bash
# Test different thresholds
yolo val model=best.pt data=data.yaml conf=0.20  # Lower (better recall)
yolo val model=best.pt data=data.yaml conf=0.15  # Even lower
yolo val model=best.pt data=data.yaml conf=0.30  # Higher (better precision)
```

**Fix - Option B: Add more diverse training examples:**
- Small potholes
- Potholes in shadows
- Wet roads
- Different lighting conditions

#### 5. Model Path Error

**Error:**
```
FileNotFoundError: 'scripts/runs/yolov8n_stage1/weights/best.pt'
```

**Actual path:**
```
runs/detect/scripts/runs/yolov8n_stage1/weights/best.pt
```

**Fix:** This is already fixed in the latest version. Pull the latest changes:
```bash
git pull origin claude/add-pothole-detection-660lv
```

### Quick Diagnostic Checklist

Run these commands to diagnose your training issues:

```bash
# 1. Check dataset size
echo "Train images: $(ls data/pothole_crack_detection/images/train/ | wc -l)"
echo "Val images: $(ls data/pothole_crack_detection/images/val/ | wc -l)"

# Target: 3,000+ training images
# If < 3,000: Download more datasets!

# 2. Check number of classes
cat data/pothole_crack_detection/data.yaml | grep "nc:"

# Should show: nc: 4
# If nc: 1 → Missing crack detection!

# 3. Check training metrics
tail -20 scripts/runs/yolov8n_stage2/results.csv

# Look for mAP@50(B) column
# If stuck around 0.7-0.75 for last 10 epochs → Plateau (need more data)

# 4. Check class distribution
python -c "
import glob
from collections import Counter
labels = glob.glob('data/pothole_crack_detection/labels/train/*.txt')
classes = Counter()
for lbl in labels:
    with open(lbl) as f:
        for line in f:
            cls = int(line.split()[0])
            classes[cls] += 1
print('Class distribution:', dict(classes))
"
# Should show balanced distribution across 4 classes
```

### Performance Expectations

| Dataset Size | Classes | Expected mAP@50 | Expected Recall |
|--------------|---------|-----------------|-----------------|
| 1,000-2,000  | 1       | 65-75%          | 55-65%          |
| 2,000-3,000  | 1       | 70-77%          | 60-70%          |
| 3,000-4,000  | 4       | 82-87%          | 72-78%          |
| 4,000-5,000  | 4       | 87-92%          | 77-83%          |
| 5,000+       | 4       | 90-95%          | 80-88%          |

**Current Issue:** Most users have 2,000-3,000 images with 1 class → 72-77% mAP@50

**Target:** 4,000-5,000 images with 4 classes → 87-92% mAP@50

### When to Retrain vs. Fine-Tune

**Retrain from scratch if:**
- ✅ You added significantly more data (doubled dataset size)
- ✅ You added new classes (e.g., added crack detection)
- ✅ Current mAP@50 < 75%

**Fine-tune existing model if:**
- ✅ Current mAP@50 > 80%
- ✅ Only added 10-20% more data
- ✅ Same classes, just more examples

```bash
# Fine-tune from existing model
python scripts/yolov8_detection/train_yolov8.py \
    --resume_stage1 runs/detect/scripts/runs/yolov8n_stage1/weights/best.pt \
    --skip_stage1 \
    --epochs_stage2 20  # Fewer epochs for fine-tuning
```

## 📞 Support

- **Documentation**: `scripts/yolov8_detection/README.md`
- **Ultralytics Docs**: https://docs.ultralytics.com/
- **Training Issues**: Check `scripts/runs/yolov8n_stage2/train.log`
- **Inference Issues**: Run with `--device cpu` to test if it's GPU-related
- **Low Metrics Issues**: See Troubleshooting section above

## ✅ Success Criteria

After completing all steps, you should have:
- ✅ Trained model with mAP@50 > 90%
- ✅ Real-time inference at 30+ FPS
- ✅ Accurate pothole and crack detection
- ✅ Significantly better than old EfficientNet model
- ✅ Working ROI filtering

---

**Next Steps**: Start with Step 1 (Install Dependencies) and work through each step sequentially!
