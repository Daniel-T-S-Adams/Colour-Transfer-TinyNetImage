import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

def generate_synthetic_image(generator_type: str = "gradient", 
                           size: Tuple[int, int] = (64, 64),
                           seed: Optional[int] = None,
                           **kwargs) -> np.ndarray:
    """
    Generate synthetic 64x64 RGB images with various color distributions.
    
    Args:
        generator_type: Type of synthetic image to generate
            - "gradient": Linear color gradients
            - "radial": Radial color gradients  
            - "noise": Colored noise patterns
            - "blocks": Random color blocks
            - "correlated_blocks": Spatially correlated color blocks
            - "random_walk": Random walk in RGB space
            - "spiral": Spiral color patterns
            - "checkerboard": Checkerboard patterns
        size: Image dimensions (height, width)
        seed: Random seed for reproducibility
        **kwargs: Additional parameters for specific generators
    
    Returns:
        RGB image array of shape (height, width, 3) with values in [0,1]
    """
    if seed is not None:
        np.random.seed(seed)
    
    height, width = size
    
    if generator_type == "gradient":
        return _generate_gradient(height, width, **kwargs)
    elif generator_type == "radial":
        return _generate_radial(height, width, **kwargs)
    elif generator_type == "noise":
        return _generate_noise(height, width, **kwargs)
    elif generator_type == "blocks":
        return _generate_blocks(height, width, **kwargs)
    elif generator_type == "correlated_blocks":
        return _generate_correlated_blocks(height, width, **kwargs)
    elif generator_type == "random_walk":
        return _generate_random_walk(height, width, **kwargs)
    elif generator_type == "spiral":
        return _generate_spiral(height, width, **kwargs)
    elif generator_type == "checkerboard":
        return _generate_checkerboard(height, width, **kwargs)
    else:
        raise ValueError(f"Unknown generator_type: {generator_type}")

def _generate_gradient(height: int, width: int, 
                      direction: str = "horizontal",
                      colors: Optional[Tuple] = None) -> np.ndarray:
    """Generate linear gradient images."""
    if colors is None:
        # Random start and end colors
        start_color = np.random.rand(3)
        end_color = np.random.rand(3)
    else:
        start_color, end_color = colors
    
    img = np.zeros((height, width, 3))
    
    if direction == "horizontal":
        for i in range(width):
            alpha = i / (width - 1)
            img[:, i, :] = (1 - alpha) * start_color + alpha * end_color
    elif direction == "vertical":
        for i in range(height):
            alpha = i / (height - 1)
            img[i, :, :] = (1 - alpha) * start_color + alpha * end_color
    elif direction == "diagonal":
        for i in range(height):
            for j in range(width):
                alpha = (i + j) / (height + width - 2)
                img[i, j, :] = (1 - alpha) * start_color + alpha * end_color
    
    return np.clip(img, 0, 1)

def _generate_radial(height: int, width: int,
                    center: Optional[Tuple] = None,
                    colors: Optional[Tuple] = None) -> np.ndarray:
    """Generate radial gradient images."""
    if center is None:
        center = (height // 2, width // 2)
    if colors is None:
        center_color = np.random.rand(3)
        edge_color = np.random.rand(3)
    else:
        center_color, edge_color = colors
    
    img = np.zeros((height, width, 3))
    cy, cx = center
    max_dist = np.sqrt((height//2)**2 + (width//2)**2)
    
    for i in range(height):
        for j in range(width):
            dist = np.sqrt((i - cy)**2 + (j - cx)**2)
            alpha = min(dist / max_dist, 1.0)
            img[i, j, :] = (1 - alpha) * center_color + alpha * edge_color
    
    return np.clip(img, 0, 1)

def _generate_noise(height: int, width: int,
                   noise_type: str = "gaussian",
                   scale: float = 1.0) -> np.ndarray:
    """Generate colored noise patterns."""
    if noise_type == "gaussian":
        img = np.random.normal(0.5, 0.2 * scale, (height, width, 3))
    elif noise_type == "uniform":
        img = np.random.uniform(0, scale, (height, width, 3))
    elif noise_type == "fractal":
        # Simple fractal-like noise by combining multiple scales
        img = np.zeros((height, width, 3))
        for octave in [1, 2, 4, 8]:
            h_scaled = max(1, height // octave)
            w_scaled = max(1, width // octave)
            noise = np.random.uniform(0, scale/octave, (h_scaled, w_scaled, 3))
            # Resize to full image size (simple nearest neighbor)
            noise_resized = np.repeat(np.repeat(noise, octave, axis=0), octave, axis=1)
            # Crop to exact size
            noise_resized = noise_resized[:height, :width, :]
            img += noise_resized
        img /= img.max() if img.max() > 0 else 1
    
    return np.clip(img, 0, 1)

def _generate_blocks(height: int, width: int,
                    block_size: int = 8,
                    num_colors: int = 16) -> np.ndarray:
    """Generate random color blocks."""
    # Generate random colors
    colors = np.random.rand(num_colors, 3)
    
    img = np.zeros((height, width, 3))
    
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            color_idx = np.random.randint(num_colors)
            end_i = min(i + block_size, height)
            end_j = min(j + block_size, width)
            img[i:end_i, j:end_j, :] = colors[color_idx]
    
    return img

def _generate_correlated_blocks(height: int, width: int,
                               block_size: int = 8,
                               change_prob: float = 0.15) -> np.ndarray:
    """
    Generate spatially correlated color blocks.
    
    Starting from the first block, moves horizontally with probability 'change_prob'
    of changing to a different random color from the full RGB space.
    
    Args:
        height, width: Image dimensions
        block_size: Size of each square block in pixels
        change_prob: Probability of color change when moving to next block (default: 0.15)
    """
    img = np.zeros((height, width, 3))
    
    # Calculate grid dimensions
    blocks_per_row = width // block_size
    blocks_per_col = height // block_size
    
    # Start with a random color from full RGB space
    current_color = np.random.rand(3)
    
    # Fill blocks row by row, left to right
    for row in range(blocks_per_col):
        for col in range(blocks_per_row):
            # Calculate pixel coordinates for this block
            start_i = row * block_size
            end_i = min(start_i + block_size, height)
            start_j = col * block_size
            end_j = min(start_j + block_size, width)
            
            # Fill the block with current color
            img[start_i:end_i, start_j:end_j, :] = current_color
            
            # Decide whether to change color for next block
            if np.random.random() < change_prob:
                # Change to a new random color from full RGB space
                current_color = np.random.rand(3)
            # else: keep the same color
    
    return img

def _generate_random_walk(height: int, width: int,
                         walk_prob: float = 0.97,
                         step_size: float = 0.05) -> np.ndarray:
    """
    Generate image using random walk in RGB space with block-based ordering.
    
    Starting from top-left with random color, fills pixels in 8x8 block order.
    For each subsequent pixel:
    - With probability walk_prob (97%), do random walk step in RGB space
    - With probability (1-walk_prob) (3%), pick completely random color
    
    Block ordering creates spatial correlation within blocks while allowing
    discontinuities between blocks.
    
    Args:
        height, width: Image dimensions
        walk_prob: Probability of doing random walk vs random jump (default: 0.75)
        step_size: Standard deviation of random walk step in RGB space (default: 0.05)
    """
    img = np.zeros((height, width, 3))
    
    # Start with random color in top-left
    current_color = np.random.rand(3)  # Random RGB in [0,1]
    img[0, 0, :] = current_color
    
    # Fill pixels in 8x8 block order
    block_size = 8
    blocks_per_row = width // block_size
    blocks_per_col = height // block_size
    
    first_pixel = True
    
    # Iterate through blocks (left-to-right, top-to-bottom)
    for block_row in range(blocks_per_col):
        for block_col in range(blocks_per_row):
            # Within each block, fill pixels left-to-right, top-to-bottom
            for pixel_row in range(block_size):
                for pixel_col in range(block_size):
                    # Calculate actual image coordinates
                    row = block_row * block_size + pixel_row
                    col = block_col * block_size + pixel_col
                    
                    # Skip if we've already set this pixel (first pixel)
                    if first_pixel:
                        first_pixel = False
                        continue
                    
                    # Decide: random walk or random jump?
                    if np.random.random() < walk_prob:
                        # Random walk: take a step in RGB space
                        step = np.random.normal(0, step_size, 3)  # Gaussian step
                        new_color = current_color + step
                        # Clip to [0,1] to keep valid RGB values
                        new_color = np.clip(new_color, 0, 1)
                    else:
                        # Random jump: pick completely new random color
                        new_color = np.random.rand(3)
                    
                    # Set pixel color and update current color for next iteration
                    img[row, col, :] = new_color
                    current_color = new_color
    
    return img

def _generate_spiral(height: int, width: int,
                    num_spirals: int = 3,
                    colors: Optional[list] = None) -> np.ndarray:
    """Generate spiral color patterns."""
    if colors is None:
        colors = [np.random.rand(3) for _ in range(num_spirals)]
    
    img = np.zeros((height, width, 3))
    cy, cx = height // 2, width // 2
    
    for i in range(height):
        for j in range(width):
            dx, dy = j - cx, i - cy
            angle = np.arctan2(dy, dx)
            dist = np.sqrt(dx**2 + dy**2)
            
            # Create spiral pattern
            spiral_value = (angle + dist * 0.1) % (2 * np.pi)
            color_idx = int((spiral_value / (2 * np.pi)) * len(colors))
            color_idx = min(color_idx, len(colors) - 1)
            
            img[i, j, :] = colors[color_idx]
    
    return img

def _generate_checkerboard(height: int, width: int,
                          square_size: int = 8,
                          colors: Optional[Tuple] = None) -> np.ndarray:
    """Generate checkerboard patterns."""
    if colors is None:
        color1 = np.random.rand(3)
        color2 = np.random.rand(3)
    else:
        color1, color2 = colors
    
    img = np.zeros((height, width, 3))
    
    for i in range(height):
        for j in range(width):
            square_i = i // square_size
            square_j = j // square_size
            if (square_i + square_j) % 2 == 0:
                img[i, j, :] = color1
            else:
                img[i, j, :] = color2
    
    return img

def visualize_synthetic_options(save_path: str = "outputs/synthetic_options.png"):
    """
    Generate and display examples of all synthetic image types.
    """
    generators = [
        ("gradient", {"direction": "horizontal"}),
        ("gradient", {"direction": "vertical"}),
        ("radial", {}),
        ("noise", {"noise_type": "gaussian"}),
        ("blocks", {"block_size": 8}),
                    ("correlated_blocks", {"change_prob": 0.15}),
                       ("random_walk", {"walk_prob": 0.97}),
        ("spiral", {"num_spirals": 2})
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for idx, (gen_type, kwargs) in enumerate(generators):
        row = idx // 4
        col = idx % 4
        
        # Generate synthetic image
        img = generate_synthetic_image(gen_type, seed=42, **kwargs)
        
        # Display
        axes[row, col].imshow(img, interpolation='bicubic')
        title = f"{gen_type}"
        if kwargs:
            key_param = list(kwargs.keys())[0]
            title += f"\n({key_param}={kwargs[key_param]})"
        axes[row, col].set_title(title, fontsize=10)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✅ Synthetic options visualization saved to: {save_path}")
    plt.show()
    
    return fig

if __name__ == "__main__":
    print("=== Synthetic Image Generator Demo ===")
    
    # Create outputs directory if it doesn't exist
    import os
    os.makedirs("outputs", exist_ok=True)
    
    # Generate examples
    print("Generating examples of all synthetic image types...")
    visualize_synthetic_options()
    
    # Test individual generators
    print("\nTesting individual generators...")
    for gen_type in ["gradient", "radial", "noise", "blocks", "correlated_blocks", "random_walk", "spiral", "checkerboard"]:
        img = generate_synthetic_image(gen_type, seed=42)
        print(f"✅ {gen_type}: shape={img.shape}, range=[{img.min():.3f}, {img.max():.3f}]")
    
    print("\n=== Demo Complete ===") 