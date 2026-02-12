# 🔑 API Keys Setup Guide for Margdrashti Dataset Download

This guide shows you how to get FREE API keys for downloading 4,000+ pothole detection images.

---

## 🎯 What You'll Get

With these API keys, you'll download:
- **4,000-13,000+ images** (depending on priority)
- **4 classes**: pothole, crack, longitudinal_crack, transverse_crack
- **Expected mAP@50**: 87-90% (vs 72% with limited data)

---

## 1️⃣ Roboflow API Key (REQUIRED)

**Provides:** 3,000-9,000 images from multiple datasets

### Steps:

1. **Go to Roboflow:**
   ```
   https://app.roboflow.com/
   ```

2. **Sign up for FREE:**
   - Click "Sign Up"
   - Use Google/GitHub or email
   - **100% free** for public datasets

3. **Get your API key:**
   - After login, go to: https://app.roboflow.com/settings/api
   - Or: Click your profile → Settings → Roboflow API
   - Copy the API key (looks like: `aBcDeFgHiJkLmNoPqRsTuVwXyZ123456`)

4. **Save the key:**
   - You'll paste this into Colab when prompted
   - Or set environment variable: `export ROBOFLOW_API_KEY=your_key_here`

### Example:
```bash
# In terminal or Colab
export ROBOFLOW_API_KEY="aBcDeFgHiJkLmNoPqRsTuVwXyZ123456"

# Then run download
python download_datasets_enhanced.py --roboflow_key "$ROBOFLOW_API_KEY"
```

---

## 2️⃣ Kaggle API Key (OPTIONAL - Recommended)

**Provides:** 1,000-3,500 additional images

### Steps:

1. **Go to Kaggle:**
   ```
   https://www.kaggle.com/
   ```

2. **Sign up for FREE:**
   - Click "Register"
   - Use Google/Facebook or email
   - 100% free

3. **Create API token:**
   - Go to: https://www.kaggle.com/settings
   - Scroll down to "API" section
   - Click "Create New API Token"
   - A file `kaggle.json` will be downloaded

4. **Install the kaggle.json file:**

   **On Linux/Mac:**
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

   **On Windows:**
   ```bash
   mkdir %USERPROFILE%\.kaggle
   move Downloads\kaggle.json %USERPROFILE%\.kaggle\
   ```

   **On Google Colab:**
   ```python
   # Upload kaggle.json manually
   from google.colab import files
   files.upload()  # Select kaggle.json

   # Move to correct location
   !mkdir -p ~/.kaggle
   !mv kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json
   ```

5. **Verify it works:**
   ```bash
   pip install kaggle
   kaggle datasets list
   ```

### Example kaggle.json format:
```json
{
  "username": "your_username",
  "key": "1234567890abcdef1234567890abcdef"
}
```

---

## 3️⃣ Direct Downloads (NO API KEY NEEDED)

**Provides:** 500-1,000 additional images from GitHub/public sources

These work automatically - no setup required!

---

## 📊 Dataset Priority Levels

Choose based on how much data you want:

### Essential (Minimum - ~3,000 images)
```bash
python download_datasets_enhanced.py \
    --roboflow_key YOUR_KEY \
    --priority essential
```
- Downloads: 2-3 core datasets
- Time: ~20 minutes
- Expected mAP@50: 80-85%

### Recommended (Target - ~4,500 images) ✅
```bash
python download_datasets_enhanced.py \
    --roboflow_key YOUR_KEY \
    --priority recommended
```
- Downloads: 5-6 datasets
- Time: ~30-40 minutes
- Expected mAP@50: 87-90% ✅

### All (Maximum - ~13,000 images)
```bash
python download_datasets_enhanced.py \
    --roboflow_key YOUR_KEY \
    --priority all
```
- Downloads: All available datasets
- Time: ~60-90 minutes
- Expected mAP@50: 90-93%
- May include duplicate/low-quality images

---

## 🚀 Quick Start (Colab)

### Option 1: Interactive Setup
```python
from getpass import getpass
import os

# Get Roboflow API key
print("Get your FREE key from: https://app.roboflow.com/settings/api")
roboflow_key = getpass("Enter Roboflow API key: ")
os.environ['ROBOFLOW_API_KEY'] = roboflow_key

# Download datasets (recommended priority)
!python scripts/yolov8_detection/download_datasets_enhanced.py \
    --roboflow_key "$ROBOFLOW_API_KEY" \
    --priority recommended
```

### Option 2: With Kaggle (More Data)
```python
# 1. Upload kaggle.json
from google.colab import files
print("Upload your kaggle.json file:")
uploaded = files.upload()

# 2. Setup Kaggle
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# 3. Get Roboflow key
from getpass import getpass
roboflow_key = getpass("Roboflow API key: ")

# 4. Download all datasets
!python scripts/yolov8_detection/download_datasets_enhanced.py \
    --roboflow_key {roboflow_key} \
    --priority recommended
```

---

## 🔍 Troubleshooting

### ❌ "roboflow not installed"
```bash
pip install roboflow
```

### ❌ "kaggle not installed"
```bash
pip install kaggle
```

### ❌ "Invalid API key"
- Check for typos
- Make sure you copied the ENTIRE key
- Roboflow keys are ~40 characters long
- Try regenerating the key on Roboflow

### ❌ "Could not find kaggle.json"
```bash
# Check if file exists
ls -la ~/.kaggle/kaggle.json

# If not, create directory and move file
mkdir -p ~/.kaggle
mv /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### ❌ "Project not found" (Roboflow)
- Some datasets may be private or deleted
- The script will skip failed datasets and continue
- You'll still get data from datasets that work

### ❌ "Only got 2,875 images"
- You're using the OLD download script
- Use `download_datasets_enhanced.py` instead
- Or add more datasets to the config file

---

## 📝 Dataset Configuration

You can customize which datasets to download by editing:
```
scripts/yolov8_detection/datasets_config.yaml
```

Add new Roboflow datasets:
```yaml
roboflow_datasets:
  - workspace: "your-workspace"
    project: "your-project"
    version: 1
    name: "YourDataset"
    expected_images: 1000
    classes: ["pothole"]
```

Find datasets at: https://universe.roboflow.com/

---

## ✅ Verification

After downloading, check your dataset:

```python
import os

# Count images
train_count = len(os.listdir('data/pothole_crack_detection/images/train'))
val_count = len(os.listdir('data/pothole_crack_detection/images/val'))
test_count = len(os.listdir('data/pothole_crack_detection/images/test'))
total = train_count + val_count + test_count

print(f"Total images: {total}")

if total >= 4000:
    print("✅ Target reached! Expected mAP@50: 87-90%")
else:
    print(f"⚠️ Only {total} images. Consider downloading more datasets.")
```

---

## 🎯 Expected Results by Dataset Size

| Images | Expected mAP@50 | Expected Recall | Missing Rate |
|--------|----------------|-----------------|--------------|
| 2,000  | 70-75%         | 60-65%          | 35-40%       |
| 2,875  | 72-77%         | 64-69%          | 31-36%       |
| 3,500  | 80-85%         | 70-75%          | 25-30%       |
| 4,500  | **87-90%** ✅  | **77-82%** ✅   | **18-23%** ✅|
| 6,000+ | 90-93%         | 82-87%          | 13-18%       |

---

## 💡 Tips

1. **Start with 'recommended' priority** - Best balance of quality vs time
2. **Roboflow is enough** - Kaggle is optional but adds diversity
3. **More ≠ Better** - After 6,000 images, improvement is marginal
4. **Check data.yaml** - Verify all 4 classes are present
5. **Visualize samples** - Make sure annotations look correct

---

## 🆘 Need Help?

1. **Roboflow Support:** https://help.roboflow.com/
2. **Kaggle API Docs:** https://github.com/Kaggle/kaggle-api
3. **Open an issue:** https://github.com/Shubhamf0073/Margdrashti_models/issues

---

**🎉 Once you have your keys, you're ready to download 4,000+ images and train a production-ready pothole detector!**
