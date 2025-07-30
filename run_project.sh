#!/bin/bash

# TinyNetImages Color Transfer Project Runner
# Usage: ./run_project.sh [source_index] [target_index]
# Example: ./run_project.sh 1500 5000
#
# If no arguments provided, uses defaults: source=97803 (cliff), target=78 (goldfish)

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Parse Arguments ---
SOURCE_IDX="${1:-97803}"  # Default to 97803 if not provided
TARGET_IDX="${2:-78}"     # Default to 78 if not provided

echo "=== TinyNetImages Color Transfer Project ==="
echo "Source image index: $SOURCE_IDX"
echo "Target image index: $TARGET_IDX"

# --- Environment Setup ---
VENV_DIR=".venv"
SRC_DIR="src"
PYTHON_EXEC="$VENV_DIR/bin/python3"

echo "--- Checking for Virtual Environment ---"
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found at '$VENV_DIR'."
    echo "Please set up the environment first by running:"
    echo "python3 -m venv .venv"
    echo "source .venv/bin/activate"
    echo "pip install -r requirements.txt"
    exit 1
fi

echo "✅ Virtual environment found. Activating..."
source "$VENV_DIR/bin/activate"

# --- Clean up old results (optional but recommended) ---
echo "--- Cleaning up old results ---"
rm -f data/rgb_distributions.npz data/transport_plan.npz
rm -f outputs/*.png
echo "✅ Old results cleaned up"

# --- Step 1: Download Dataset ---
echo -e "\n--- Running Step 1: Download Dataset ---"
$PYTHON_EXEC "$SRC_DIR/data_loader.py"

# --- Step 3: Build Empirical RGB Distributions ---
echo -e "\n--- Running Step 3: Build RGB Distributions ---"
echo "Using source image #$SOURCE_IDX and target image #$TARGET_IDX"
$PYTHON_EXEC "$SRC_DIR/rgb_distributions.py" "$SOURCE_IDX" "$TARGET_IDX"

# --- Step 4 & 5: Compute Cost Matrix and Transport Plan ---
echo -e "\n--- Running Steps 4 & 5: Compute Transport Plan ---"
$PYTHON_EXEC "$SRC_DIR/compute_transport_plan.py"

# --- Step 6: Perform Color Transfer ---
echo -e "\n--- Running Step 6: Perform Color Transfer ---"
$PYTHON_EXEC "$SRC_DIR/perform_color_transfer.py"

# --- Final Verification ---
echo -e "\n\n🎉🎉🎉 Color Transfer Complete! 🎉🎉🎉"
echo "Images used:"
echo "  - Source: #$SOURCE_IDX"  
echo "  - Target: #$TARGET_IDX"
echo ""
echo "Check the 'outputs/' directory for results:"
echo "  - outputs/color_transfer_comparison.png (main result)"
echo "  - outputs/recolored_source_image.png (final image only)"
echo "  - outputs/rgb_distributions.png (color analysis)"
echo ""
echo "To try different images, run:"
echo "  ./run_project.sh [source_index] [target_index]" 