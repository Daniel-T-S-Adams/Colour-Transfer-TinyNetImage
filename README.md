# TinyNetImages Color Transfer

Transfer colors between images using optimal transport and the Sinkhorn algorithm. This project takes two images and transfers the color palette from one to the other while preserving the structure.

## What It Does

🎨 **Input**: Two images (source + target)  
🚀 **Process**: Optimal transport between RGB color distributions  
✨ **Output**: Source image recolored with target's color palette  

## Quick Start

### 1. Setup Environment---Using Git bash
```bash
# Clone/download the project
# Navigate to project directory

# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate

#downlard Microsoft Visual C++ 14.0

# Install dependencies
pip install -r requirements.txt

#download dataset
python src/data_loader.py


# Make scripts executable
chmod +x run_project.sh
```

### 2. Choose Images (Optional)
```bash
# Browse random sample images to pick indices
./browse_images.py
```
This shows 12 random images with their index numbers. Pick two you like!

### 3. Run Color Transfer
```bash
# Option A: Use your chosen images
./run_project.sh [source_index] [target_index]

# Option B: Use defaults (cliff → goldfish)
./run_project.sh

# Examples:
./run_project.sh 1500 5000    # Transfer colors from image 1500 to image 5000
./run_project.sh 25432 89123  # Any valid indices work
```

## Example Workflow

```bash
# 1. Setup (one time only)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
chmod +x *.sh *.py

# 2. Browse images (optional)
./browse_images.py
# Look at image_browser.png, pick two interesting indices

# 3. Run color transfer
./run_project.sh 12345 67890

# 4. Check results
ls outputs/
# color_transfer_comparison.png  ← Main result (before/after comparison)
# recolored_source_image.png     ← Final recolored image only
# rgb_distributions.png          ← Color analysis visualization
```

## Project Structure

```
├── src/                   # Python source code
├── data/                  # Downloaded dataset & intermediate files
├── outputs/               # Generated images and results
├── run_project.sh         # Main script (runs everything)
├── browse_images.py       # Image browser helper
└── requirements.txt       # Dependencies
```

## What Happens During Execution

1. **Downloads** Tiny ImageNet dataset (~237MB)
2. **Selects** your chosen source and target images
3. **Analyzes** RGB color distributions (4096 pixels → 3D points)
4. **Computes** optimal transport plan using Sinkhorn algorithm (~30 seconds)
5. **Transfers** colors using barycentric mapping
6. **Generates** comparison visualizations

## Expected Results

- **Runtime**: ~1-2 minutes total (30s for optimal transport)
- **Memory**: ~64MB for transport computation
- **Quality**: Professional color transfer with smooth gradients

## Troubleshooting

**"No module named 'ot'"**: Make sure virtual environment is activated  
**"Index out of range"**: Choose indices between 0 and 99,999  
**"Dataset not found"**: Script will auto-download on first run  

## Technical Details

- **Algorithm**: Entropic regularized optimal transport
- **Implementation**: Python Optimal Transport (POT) library
- **Images**: 64×64 pixels from Tiny ImageNet dataset
- **Method**: Stabilized Sinkhorn with ε=0.01 regularization

---

🎯 **TL;DR**: Run `./run_project.sh` for defaults, or `./browse_images.py` then `./run_project.sh [src] [tgt]` for custom images! 