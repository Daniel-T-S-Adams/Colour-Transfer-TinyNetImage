#!/bin/bash

# TinyNetImages Color Transfer Project Runner
# Usage: 
#   ./run_project.sh [source_index] [target_index]           # Use dataset images
#   ./run_project.sh synthetic [source_type] [target_type]   # Use synthetic images
#
# Examples:
#   ./run_project.sh 1500 5000                              # Dataset images
#   ./run_project.sh synthetic gradient radial              # Synthetic images  
#   ./run_project.sh synthetic                              # Synthetic with defaults
#
# If no arguments provided, uses defaults: source=97803 (cliff), target=78 (goldfish)

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Parse Arguments ---
SEED="42"  # Default seed

if [ "$1" = "synthetic" ]; then
    USE_SYNTHETIC=true
    SOURCE_TYPE="${2:-gradient}"  # Default to gradient
    TARGET_TYPE="${3:-radial}"    # Default to radial
    
    # Check for seed parameter
    for i in "$@"; do
        if [[ $i == --seed=* ]]; then
            SEED="${i#*=}"
        fi
    done
    
    # Check for --seed followed by value (simpler approach)
    for i in "$@"; do
        if [[ $i == "--seed" ]]; then
            # Find the next argument after --seed
            for j in "$@"; do
                if [[ $j =~ ^[0-9]+$ ]] && [[ $j != "$SEED" ]]; then
                    SEED="$j"
                    break
                fi
            done
        fi
    done
    
    echo "=== TinyNetImages Color Transfer Project (Synthetic Mode) ==="
    echo "Source type: $SOURCE_TYPE"
    echo "Target type: $TARGET_TYPE"
    echo "Seed: $SEED"
else
    USE_SYNTHETIC=false
    SOURCE_IDX="${1:-97803}"  # Default to 97803 if not provided
    TARGET_IDX="${2:-78}"     # Default to 78 if not provided
    echo "=== TinyNetImages Color Transfer Project (Dataset Mode) ==="
    echo "Source image index: $SOURCE_IDX"
    echo "Target image index: $TARGET_IDX"
fi

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

# --- Step 1: Download Dataset (only needed for dataset mode) ---
if [ "$USE_SYNTHETIC" = false ]; then
    echo -e "\n--- Running Step 1: Download Dataset ---"
    $PYTHON_EXEC "$SRC_DIR/data_loader.py"
fi

# --- Step 3: Build Empirical RGB Distributions ---
echo -e "\n--- Running Step 3: Build RGB Distributions ---"
if [ "$USE_SYNTHETIC" = true ]; then
    echo "Using synthetic images: source=$SOURCE_TYPE, target=$TARGET_TYPE"
    $PYTHON_EXEC "$SRC_DIR/rgb_distributions.py" synthetic "$SOURCE_TYPE" "$TARGET_TYPE" --seed "$SEED"
else
    echo "Using dataset images: source=#$SOURCE_IDX, target=#$TARGET_IDX"
    $PYTHON_EXEC "$SRC_DIR/rgb_distributions.py" "$SOURCE_IDX" "$TARGET_IDX"
fi

# --- Step 4 & 5: Compute Cost Matrix and Transport Plan ---
echo -e "\n--- Running Steps 4 & 5: Compute Transport Plan ---"
$PYTHON_EXEC "$SRC_DIR/compute_transport_plan.py"

# --- Step 6: Perform Color Transfer ---
echo -e "\n--- Running Step 6: Perform Color Transfer ---"
$PYTHON_EXEC "$SRC_DIR/perform_color_transfer.py"

# --- Final Verification ---
echo -e "\n\n🎉🎉🎉 Color Transfer Complete! 🎉🎉🎉"
if [ "$USE_SYNTHETIC" = true ]; then
    echo "Synthetic images used:"
    echo "  - Source: $SOURCE_TYPE"  
    echo "  - Target: $TARGET_TYPE"
    echo "  - Seed: $SEED"
else
    echo "Dataset images used:"
    echo "  - Source: #$SOURCE_IDX"  
    echo "  - Target: #$TARGET_IDX"
fi
echo ""
echo "Check the 'outputs/' directory for results:"
echo "  - outputs/color_transfer_comparison.png (main result)"
echo "  - outputs/recolored_source_image.png (final image only)"
echo "  - outputs/rgb_distributions.png (color analysis)"
echo "  - outputs/selected_images.png (source and target images)"
echo ""
echo "Usage examples:"
echo "  ./run_project.sh [source_index] [target_index]           # Dataset images"
echo "  ./run_project.sh synthetic [source_type] [target_type]   # Synthetic images"
echo "  ./run_project.sh synthetic [source_type] [target_type] --seed [seed]  # Custom seed"
echo ""
echo "Synthetic types: gradient, radial, noise, blocks, spiral, checkerboard" 