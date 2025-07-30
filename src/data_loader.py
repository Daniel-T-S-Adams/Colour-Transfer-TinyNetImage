import os
import requests
import zipfile
from tqdm import tqdm
import numpy as np
from PIL import Image

def download_and_unzip(url, target_path):
    """
    Downloads and unzips the Tiny ImageNet dataset.
    """
    if os.path.exists(target_path):
        print("Dataset already downloaded.")
        return
    
    # Create parent directory if target_path has one
    parent_dir = os.path.dirname(target_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    
    # Download the file
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    zip_path = f"{target_path}.zip"
    
    with open(zip_path, 'wb') as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            f.write(data)
            
    # Unzip the file
    print(f"Unzipping {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc='Extracting '):
            extract_to = os.path.dirname(target_path) if os.path.dirname(target_path) else '.'
            zip_ref.extract(member, extract_to)
            
    os.remove(zip_path)
    print(f"Dataset downloaded and extracted to {target_path}")

def load_tiny_imagenet_image(path: str) -> np.ndarray:
    """
    Loads an image from the Tiny ImageNet dataset and returns it as a numpy array.
    """
    img = Image.open(path)
    
    # Ensure image is 64x64
    if img.size != (64, 64):
        img = img.resize((64, 64))
        
    # Ensure image has 3 channels
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    return img_array

if __name__ == "__main__":
    # Set fixed random seed for reproducibility
    np.random.seed(42)
    
    # URL for Tiny ImageNet dataset
    DATASET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    DATASET_ROOT = "data/tiny-imagenet-200"
    
    # Download and unzip the dataset
    download_and_unzip(DATASET_URL, DATASET_ROOT)
    
    print(f"Dataset root: {os.path.abspath(DATASET_ROOT)}")
    
    # Example of loading an image
    # Note: You might need to adjust the path based on the unzipped structure
    example_image_path = os.path.join(DATASET_ROOT, "train", "n01443537", "images", "n01443537_0.JPEG")
    if os.path.exists(example_image_path):
        image_array = load_tiny_imagenet_image(example_image_path)
        print(f"Example image shape: {image_array.shape}")
        print(f"Example image dtype: {image_array.dtype}")
        print(f"Example image min/max: {image_array.min()}/{image_array.max()}")
    else:
        print(f"Example image not found at: {example_image_path}")
        print("Please check the unzipped folder structure.") 