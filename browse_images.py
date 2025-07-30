#!/usr/bin/env python3

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import load_tiny_imagenet_image

def browse_random_images(num_images=12, seed=42):
    """
    Display a grid of random images with their indices to help users choose.
    
    Args:
        num_images: Number of random images to display
        seed: Random seed for reproducible results
    """
    print("=== Image Browser ===")
    print("This tool helps you pick source and target images for color transfer.")
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Get all training images
    all_images = []
    for root, dirs, files in os.walk('data/tiny-imagenet-200/train'):
        for file in files:
            if file.endswith('.JPEG'):
                all_images.append(os.path.join(root, file))
    all_images.sort()
    
    print(f"Total images available: {len(all_images)}")
    print(f"Showing {num_images} random samples...\n")
    
    # Pick random images
    random_indices = random.sample(range(len(all_images)), num_images)
    random_indices.sort()
    
    # Create visualization
    cols = 4
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]
    
    for i, idx in enumerate(random_indices):
        row = i // cols
        col = i % cols
        
        img_path = all_images[idx]
        img = load_tiny_imagenet_image(img_path)
        
        # Get class name
        class_id = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
        class_name = get_class_name(class_id)
        
        # Display image
        if row < len(axes) and col < len(axes[row]):
            axes[row][col].imshow(img, interpolation='bicubic')
            axes[row][col].set_title(f'Index: {idx}\n{class_name}', fontsize=10)
            axes[row][col].axis('off')
            
        print(f"Index {idx:5}: {class_name}")
    
    # Hide empty subplots
    for i in range(num_images, rows * cols):
        row = i // cols
        col = i % cols
        if row < len(axes) and col < len(axes[row]):
            axes[row][col].axis('off')
    
    plt.tight_layout()
    plt.savefig('image_browser.png', dpi=200, bbox_inches='tight')
    print(f"\n✅ Browser saved to: image_browser.png")
    
    print("\n" + "="*50)
    print("HOW TO USE:")
    print("1. Look at the images above and pick two indices you like")
    print("2. Run: ./run_project.sh [source_index] [target_index]")
    print("   Example: ./run_project.sh 15234 67890")
    print("\nCurrent defaults:")
    print("  - Source: 97803 (cliff)")  
    print("  - Target: 78 (goldfish)")
    print("="*50)
    
    plt.show()
    
    return random_indices

def get_class_name(class_id):
    """Get human-readable class name from class ID."""
    try:
        with open('data/tiny-imagenet-200/words.txt', 'r') as f:
            for line in f:
                if line.startswith(class_id):
                    name = line.split('\t')[1].strip()
                    # Truncate long names
                    if len(name) > 20:
                        name = name[:17] + "..."
                    return name
    except:
        pass
    return class_id[:8] + "..."

if __name__ == "__main__":
    # Check if dataset exists
    if not os.path.exists('data/tiny-imagenet-200'):
        print("❌ Dataset not found!")
        print("Please run: python src/data_loader.py first")
        exit(1)
    
    browse_random_images(12) 