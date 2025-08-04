#!/usr/bin/env python3

import os
import sys
import matplotlib.pyplot as plt

# Add src to path so we can import our modules
sys.path.append('src')
from synthetic_generator import generate_synthetic_image, visualize_synthetic_options

def main():
    """
    Display examples of all synthetic image types to help users choose.
    """
    print("=== Synthetic Image Browser ===")
    print("This tool helps you pick synthetic image types for color transfer.")
    
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    # Generate examples
    print("Generating examples of all synthetic image types...")
    visualize_synthetic_options()
    
    print("\n" + "="*60)
    print("SYNTHETIC IMAGE TYPES AVAILABLE:")
    print("="*60)
    
    types_info = [
        ("gradient", "Linear color gradients (horizontal, vertical, diagonal)"),
        ("radial", "Radial gradients from center to edges"),
        ("noise", "Colored noise patterns (gaussian, uniform, fractal)"),
        ("blocks", "Random colored rectangular blocks"),
                    ("correlated_blocks", "Spatially correlated colored blocks (15% change probability)"),
                    ("random_walk", "Random walk in RGB space (97% walk, 3% jump probability)"),
        ("spiral", "Spiral color patterns"),
        ("checkerboard", "Checkerboard patterns with two colors")
    ]
    
    for i, (type_name, description) in enumerate(types_info, 1):
        print(f"{i}. {type_name:12} - {description}")
    
    print("="*60)
    print("HOW TO USE:")
    print("1. Look at outputs/synthetic_options.png to see examples")
    print("2. Run: ./run_project.sh synthetic [source_type] [target_type]")
    print("   Examples:")
    print("     ./run_project.sh synthetic gradient radial")
    print("     ./run_project.sh synthetic noise spiral")
    print("     ./run_project.sh synthetic blocks correlated_blocks")
    print("     ./run_project.sh synthetic correlated_blocks checkerboard")
    print("\nCurrent defaults:")
    print("  - Source: gradient")  
    print("  - Target: radial")
    print("="*60)

if __name__ == "__main__":
    main() 