import os
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_tiny_imagenet_image

def pick_specific_images():
    """
    Pick two specific images as defined in the project instructions:
    - Source image: #97803
    - Target image: #78
    """
    # Set fixed random seed for reproducibility (as required in instructions)
    np.random.seed(42)
    
    # Read the sorted list of all training images
    with open('all_train_images.txt', 'r') as f:
        all_images = [line.strip() for line in f.readlines()]
    
    # Pick specific images by index (0-based indexing)
    target_path = all_images[78]    # Image #78 as target
    source_path = all_images[97803] # Image #97803 as source
    
    print(f"Source image (#97803): {source_path}")
    print(f"Target image (#78): {target_path}")
    
    # Load the images
    img_src = load_tiny_imagenet_image(source_path)
    img_tgt = load_tiny_imagenet_image(target_path)
    
    print(f"Source image shape: {img_src.shape}, dtype: {img_src.dtype}")
    print(f"Target image shape: {img_tgt.shape}, dtype: {img_tgt.dtype}")
    print(f"Source image range: [{img_src.min():.3f}, {img_src.max():.3f}]")
    print(f"Target image range: [{img_tgt.min():.3f}, {img_tgt.max():.3f}]")
    
    return img_src, img_tgt, source_path, target_path

def visualize_images(img_src, img_tgt, source_path, target_path, save_path="selected_images.png"):
    """
    Visualize both images side by side and save to disk.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display source image
    axes[0].imshow(img_src)
    axes[0].set_title(f'Source Image (#97803)\n{os.path.basename(source_path)}', fontsize=12)
    axes[0].axis('off')
    
    # Display target image  
    axes[1].imshow(img_tgt)
    axes[1].set_title(f'Target Image (#78)\n{os.path.basename(target_path)}', fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    
    # Also show the plot
    plt.show()
    
    return fig

if __name__ == "__main__":
    print("=== Step 2: Picking Two Images ===")
    
    # Pick the specific images
    img_src, img_tgt, source_path, target_path = pick_specific_images()
    
    # Visualize and save
    fig = visualize_images(img_src, img_tgt, source_path, target_path)
    
    print("\n=== Image Selection Complete ===")
    print("✅ Two images loaded successfully")
    print("✅ Images visualized and saved to disk")
    print(f"✅ Source image: {source_path}")
    print(f"✅ Target image: {target_path}") 