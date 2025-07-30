import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_tiny_imagenet_image
import time

def load_all_data():
    """Loads all necessary data from previous steps."""
    print("--- Loading All Necessary Data ---")
    try:
        # Load transport plan from the data directory
        transport_data = np.load('data/transport_plan.npz')
        P = transport_data['P']
        
        # Load RGB distributions from the data directory
        rgb_data = np.load('data/rgb_distributions.npz', allow_pickle=True)
        X_src = rgb_data['X_src']
        X_tgt = rgb_data['X_tgt']
        w_src = rgb_data['w_src']
        source_path = str(rgb_data['source_path'])
        target_path = str(rgb_data['target_path'])

        # Load the original images for visualization
        img_src_original = load_tiny_imagenet_image(source_path)
        img_tgt_original = load_tiny_imagenet_image(target_path)

        print("✅ All data loaded successfully.")
        print(f"   Transport plan `P` shape: {P.shape}")
        print(f"   Source points `X_src` shape: {X_src.shape}")
        print(f"   Target points `X_tgt` shape: {X_tgt.shape}")
        
        return P, X_src, X_tgt, w_src, img_src_original, img_tgt_original
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find a required data file: {e.filename}")
        print("Please ensure 'rgb_distributions.py' and 'compute_transport_plan.py' have been run.")
        return None, None, None, None, None, None

def apply_barycentric_mapping(P, X_tgt, w_src):
    """
    Applies barycentric mapping to transfer color.
    This completes the core task of Step 6.
    
    Formula: recolored_pixel_i = sum_j(P_ij * target_pixel_j) / sum_j(P_ij)
    Since sum_j(P_ij) = w_src_i = 1/N, this simplifies to N * sum_j(P_ij * target_pixel_j)
    This is equivalent to a matrix multiplication: N * (P @ X_tgt)
    """
    print("\n--- Step 6: Applying Barycentric Mapping ---")
    start_time = time.time()
    
    N = X_tgt.shape[0]
    
    # The matrix multiplication P @ X_tgt computes the numerator sum_j(P_ij * X_tgt_j) for each i.
    recolored_X_src = P @ X_tgt
    
    # The denominator for each pixel i is P.sum(axis=1), which is w_src.
    # We need to divide each row of the result by the corresponding weight.
    # Using np.newaxis ensures broadcasting works correctly.
    recolored_X_src /= w_src[:, np.newaxis]
    
    # An alternative, more direct way since w_src is uniform (1/N):
    # recolored_X_src = N * (P @ X_tgt)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"✅ Barycentric mapping applied successfully.")
    print(f"   Time taken: {duration:.4f} seconds")
    
    # Validate the output
    print(f"   Shape of recolored points: {recolored_X_src.shape}")
    print(f"   Value range before clipping: [{recolored_X_src.min():.4f}, {recolored_X_src.max():.4f}]")
    
    # Clip results to the valid [0, 1] color range
    recolored_X_src_clipped = np.clip(recolored_X_src, 0, 1)
    print(f"   Value range after clipping:  [{recolored_X_src_clipped.min():.4f}, {recolored_X_src_clipped.max():.4f}]")
    
    return recolored_X_src_clipped

def reshape_and_save_images(recolored_X_src, img_src_original, img_tgt_original):
    """Reshapes the recolored points back to an image and saves all results."""
    print("\n--- Reshaping and Saving Final Images ---")
    
    # Reshape the (4096, 3) points back to a (64, 64, 3) image
    recolored_image = recolored_X_src.reshape(64, 64, 3)
    print(f"✅ Recolored points reshaped to image: {recolored_image.shape}")
    
    # --- Create Comparison Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original Source
    axes[0].imshow(img_src_original, interpolation='bicubic')
    axes[0].set_title("Original Source\n(Cliff)")
    axes[0].axis('off')
    
    # Original Target
    axes[1].imshow(img_tgt_original, interpolation='bicubic')
    axes[1].set_title("Original Target\n(Goldfish)")
    axes[1].axis('off')
    
    # Recolored Source
    axes[2].imshow(recolored_image, interpolation='bicubic')
    axes[2].set_title("Recolored Source")
    axes[2].axis('off')

    plt.tight_layout()
    
    # Save the comparison figure to the outputs directory
    comparison_filepath = 'outputs/color_transfer_comparison.png'
    plt.savefig(comparison_filepath, dpi=300)
    print(f"✅ Comparison visualization saved to '{comparison_filepath}'")

    # Save the final recolored image to the outputs directory
    recolored_filepath = 'outputs/recolored_source_image.png'
    plt.imsave(recolored_filepath, recolored_image)
    print(f"✅ Final recolored image saved to '{recolored_filepath}'")
    
    plt.show()

if __name__ == "__main__":
    # --- Step 1: Load all required data from previous steps ---
    P, X_src, X_tgt, w_src, img_src_original, img_tgt_original = load_all_data()
    
    if P is not None:
        # --- Step 2: Apply the barycentric mapping (Core of Step 6) ---
        recolored_X_src = apply_barycentric_mapping(P, X_tgt, w_src)
        
        # --- Step 3: Reshape, visualize, and save the final results ---
        reshape_and_save_images(recolored_X_src, img_src_original, img_tgt_original)
        
        print("\n🎉 Step 6 completed successfully!") 