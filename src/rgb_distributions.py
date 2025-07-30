import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from data_loader import load_tiny_imagenet_image

def load_selected_images(source_idx=97803, target_idx=78):
    """
    Load the two selected images by index.
    
    Args:
        source_idx: Index of source image (default: 97803 - cliff)
        target_idx: Index of target image (default: 78 - goldfish)
    """
    # Set fixed random seed for reproducibility (as required in instructions)
    np.random.seed(42)
    
    # Create image list by searching in the correct data directory
    all_images = []
    for root, dirs, files in os.walk('data/tiny-imagenet-200/train'):
        for file in files:
            if file.endswith('.JPEG'):
                all_images.append(os.path.join(root, file))
    all_images.sort()
    
    # Validate indices
    if source_idx >= len(all_images) or target_idx >= len(all_images):
        print(f"❌ Error: Index out of range. Available indices: 0 to {len(all_images)-1}")
        sys.exit(1)
    
    # Pick specific images by index
    target_path = all_images[target_idx]    # Target image
    source_path = all_images[source_idx]    # Source image
    
    print(f"Loading source image #{source_idx}: {source_path}")
    print(f"Loading target image #{target_idx}: {target_path}")
    
    # Load the images
    img_src = load_tiny_imagenet_image(source_path)
    img_tgt = load_tiny_imagenet_image(target_path)
    
    return img_src, img_tgt, source_path, target_path

def build_empirical_rgb_distributions(img_src, img_tgt):
    """
    Build empirical RGB distributions from the two images.
    
    Args:
        img_src: Source image array (64,64,3) with values in [0,1]
        img_tgt: Target image array (64,64,3) with values in [0,1]
    
    Returns:
        X_src: Source RGB points (N,3) where N=4096
        w_src: Source weights (N,) uniform distribution  
        X_tgt: Target RGB points (N,3) where N=4096
        w_tgt: Target weights (N,) uniform distribution
    """
    print("\n=== Step 3: Building Empirical RGB Distributions ===")
    
    # Verify input shapes and ranges
    assert img_src.shape == (64, 64, 3), f"Source image shape should be (64,64,3), got {img_src.shape}"
    assert img_tgt.shape == (64, 64, 3), f"Target image shape should be (64,64,3), got {img_tgt.shape}"
    assert 0 <= img_src.min() and img_src.max() <= 1, f"Source image should be in [0,1], got [{img_src.min()}, {img_src.max()}]"
    assert 0 <= img_tgt.min() and img_tgt.max() <= 1, f"Target image should be in [0,1], got [{img_tgt.min()}, {img_tgt.max()}]"
    
    print(f"✅ Input validation passed")
    print(f"Source image range: [{img_src.min():.3f}, {img_src.max():.3f}]")
    print(f"Target image range: [{img_tgt.min():.3f}, {img_tgt.max():.3f}]")
    
    # Reshape images to (N, 3) where N = 64*64 = 4096
    N = 64 * 64  # 4096 pixels
    X_src = img_src.reshape(N, 3)  # (4096, 3)
    X_tgt = img_tgt.reshape(N, 3)  # (4096, 3)
    
    print(f"✅ Reshaped images: {img_src.shape} -> {X_src.shape}")
    
    # RGB values are already normalized to [0,1] by load_tiny_imagenet_image
    print(f"✅ RGB already normalized to [0,1]")
    
    # Define uniform weights: w = np.ones(N)/N for both images
    w_src = np.ones(N) / N  # Uniform distribution (4096,)
    w_tgt = np.ones(N) / N  # Uniform distribution (4096,)
    
    print(f"✅ Created uniform weights: shape {w_src.shape}, sum = {w_src.sum():.6f}")
    
    # Verify outputs
    assert X_src.shape == (N, 3), f"X_src shape should be ({N},3), got {X_src.shape}"
    assert X_tgt.shape == (N, 3), f"X_tgt shape should be ({N},3), got {X_tgt.shape}"
    assert w_src.shape == (N,), f"w_src shape should be ({N},), got {w_src.shape}"
    assert w_tgt.shape == (N,), f"w_tgt shape should be ({N},), got {w_tgt.shape}"
    assert abs(w_src.sum() - 1.0) < 1e-10, f"w_src should sum to 1, got {w_src.sum()}"
    assert abs(w_tgt.sum() - 1.0) < 1e-10, f"w_tgt should sum to 1, got {w_tgt.sum()}"
    
    print(f"✅ Output validation passed")
    print(f"X_src shape: {X_src.shape}, dtype: {X_src.dtype}")
    print(f"X_tgt shape: {X_tgt.shape}, dtype: {X_tgt.dtype}")
    print(f"w_src shape: {w_src.shape}, sum: {w_src.sum():.10f}")
    print(f"w_tgt shape: {w_tgt.shape}, sum: {w_tgt.sum():.10f}")
    
    return X_src, w_src, X_tgt, w_tgt

def visualize_rgb_distributions(X_src, X_tgt, save_path="outputs/rgb_distributions.png"):
    """
    Visualize the RGB distributions as 3D scatter plots and histograms.
    """
    print(f"\n=== Visualizing RGB Distributions ===")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 3D scatter plots
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(X_src[:, 0], X_src[:, 1], X_src[:, 2], c=X_src, s=1, alpha=0.6)
    ax1.set_xlabel('Red')
    ax1.set_ylabel('Green')
    ax1.set_zlabel('Blue')
    ax1.set_title('Source RGB Distribution\n(3D Scatter)')
    
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.scatter(X_tgt[:, 0], X_tgt[:, 1], X_tgt[:, 2], c=X_tgt, s=1, alpha=0.6)
    ax2.set_xlabel('Red')
    ax2.set_ylabel('Green')
    ax2.set_zlabel('Blue')
    ax2.set_title('Target RGB Distribution\n(3D Scatter)')
    
    # 2D projections
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(X_src[:, 0], X_src[:, 1], c=X_src, s=1, alpha=0.6)
    ax3.set_xlabel('Red')
    ax3.set_ylabel('Green')
    ax3.set_title('Source RG Projection')
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(X_tgt[:, 0], X_tgt[:, 1], c=X_tgt, s=1, alpha=0.6)
    ax4.set_xlabel('Red')
    ax4.set_ylabel('Green')
    ax4.set_title('Target RG Projection')
    ax4.grid(True, alpha=0.3)
    
    # Histograms per channel
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(X_src[:, 0], bins=50, alpha=0.7, label='Source Red', color='red')
    ax5.hist(X_src[:, 1], bins=50, alpha=0.7, label='Source Green', color='green')
    ax5.hist(X_src[:, 2], bins=50, alpha=0.7, label='Source Blue', color='blue')
    ax5.set_xlabel('Color Value')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Source RGB Histograms')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.hist(X_tgt[:, 0], bins=50, alpha=0.7, label='Target Red', color='red')
    ax6.hist(X_tgt[:, 1], bins=50, alpha=0.7, label='Target Green', color='green')
    ax6.hist(X_tgt[:, 2], bins=50, alpha=0.7, label='Target Blue', color='blue')
    ax6.set_xlabel('Color Value')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Target RGB Histograms')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ RGB distribution visualization saved to: {save_path}")
    plt.show()
    
    return fig

def print_distribution_statistics(X_src, w_src, X_tgt, w_tgt):
    """
    Print detailed statistics about the RGB distributions.
    """
    print(f"\n=== Distribution Statistics ===")
    
    # Basic statistics
    print(f"Dataset size: N = {len(X_src)} pixels per image")
    print(f"RGB space: [0,1]³")
    print(f"Weight type: Uniform (1/N = {1/len(X_src):.6f} per pixel)")
    
    # Per-channel statistics for source
    print(f"\nSource Image RGB Statistics:")
    for i, channel in enumerate(['Red', 'Green', 'Blue']):
        mean_val = np.average(X_src[:, i], weights=w_src)
        var_val = np.average((X_src[:, i] - mean_val)**2, weights=w_src)
        print(f"  {channel:5}: mean={mean_val:.3f}, std={np.sqrt(var_val):.3f}, range=[{X_src[:, i].min():.3f}, {X_src[:, i].max():.3f}]")
    
    # Per-channel statistics for target
    print(f"\nTarget Image RGB Statistics:")
    for i, channel in enumerate(['Red', 'Green', 'Blue']):
        mean_val = np.average(X_tgt[:, i], weights=w_tgt)
        var_val = np.average((X_tgt[:, i] - mean_val)**2, weights=w_tgt)
        print(f"  {channel:5}: mean={mean_val:.3f}, std={np.sqrt(var_val):.3f}, range=[{X_tgt[:, i].min():.3f}, {X_tgt[:, i].max():.3f}]")
    
    # Distance between distributions (rough estimate)
    src_mean = np.average(X_src, weights=w_src, axis=0)
    tgt_mean = np.average(X_tgt, weights=w_tgt, axis=0)
    mean_distance = np.linalg.norm(src_mean - tgt_mean)
    print(f"\nDistance between distribution means: {mean_distance:.3f}")
    print(f"Source mean RGB: [{src_mean[0]:.3f}, {src_mean[1]:.3f}, {src_mean[2]:.3f}]")
    print(f"Target mean RGB: [{tgt_mean[0]:.3f}, {tgt_mean[1]:.3f}, {tgt_mean[2]:.3f}]")

if __name__ == "__main__":
    print("=== Step 3: Build Empirical RGB Distributions ===")
    
    # Parse command line arguments
    source_idx = 97803 # Default to 97803 (cliff)
    target_idx = 78   # Default to 78 (goldfish)
    
    if len(sys.argv) > 1:
        try:
            source_idx = int(sys.argv[1])
        except ValueError:
            print(f"❌ Error: Source index '{sys.argv[1]}' is not a valid integer.")
            sys.exit(1)
    if len(sys.argv) > 2:
        try:
            target_idx = int(sys.argv[2])
        except ValueError:
            print(f"❌ Error: Target index '{sys.argv[2]}' is not a valid integer.")
            sys.exit(1)
    
    # Load the selected images
    img_src, img_tgt, source_path, target_path = load_selected_images(source_idx, target_idx)
    
    # Build empirical RGB distributions
    X_src, w_src, X_tgt, w_tgt = build_empirical_rgb_distributions(img_src, img_tgt)
    
    # Print statistics
    print_distribution_statistics(X_src, w_src, X_tgt, w_tgt)
    
    # Visualize distributions
    fig = visualize_rgb_distributions(X_src, X_tgt)
    
    print(f"\n=== Step 3 Complete ===")
    print("✅ Empirical RGB distributions built successfully")
    print("✅ Uniform weights created and validated")
    print("✅ Statistics computed and visualizations saved")
    print("✅ Ready for Step 4: Cost Matrix Computation")
    
    # Save results for next steps
    np.savez('data/rgb_distributions.npz', 
             X_src=X_src, w_src=w_src, X_tgt=X_tgt, w_tgt=w_tgt,
             source_path=source_path, target_path=target_path)
    print("✅ Data saved to data/rgb_distributions.npz for next steps")
