"""
visualize_augmentation.py
--------------------------------
This script demonstrates how the augmentation pipeline in augmentation.py
transforms a sample breast ultrasound image. It displays multiple augmented
versions of the same image to showcase the effect of data augmentation.
"""

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import os
import kagglehub

# Import the augmentation transform from augmentation.py
from augmentation import get_augmented_transform

# Download dataset using kagglehub
path = kagglehub.dataset_download("aryashah2k/breast-ultrasound-images-dataset")
DATASET_PATH = os.path.join(path, "Dataset_BUSI_with_GT")

# Ensure the dataset path exists
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")

# List available categories
categories = os.listdir(DATASET_PATH)
print("Available categories:", categories)

# Choose one category: "normal", "benign", or "malignant"
chosen_category = "malignant"  
category_path = os.path.join(DATASET_PATH, chosen_category)

if not os.path.exists(category_path):
    raise FileNotFoundError(f"Category path not found: {category_path}")

# Gather images in this category (PNG files)
images = [f for f in os.listdir(category_path) if f.lower().endswith(".png") and "_mask" not in f]
if len(images) == 0:
    raise FileNotFoundError(f"No images found in {category_path}")

# Pick any image from this list
sample_image_path = os.path.join(category_path, images[0])
print("Using sample image:", sample_image_path)

def load_sample_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

def visualize_augmentation(image_path, num_augmented=4):
    # Load original image
    original_image = load_sample_image(image_path)

    # Get augmentation transform
    transform = get_augmented_transform()

    # Generate augmented images
    augmented_images = [transform(original_image) for _ in range(num_augmented)]
    augmented_images = [transforms.ToPILImage()(img) for img in augmented_images]

    # Plot them
    fig, axes = plt.subplots(1, num_augmented + 1, figsize=(12, 4))
    
    # Original Image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Augmented Images
    for i, img in enumerate(augmented_images):
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(f"Augmented {i + 1}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.show()

# Execute visualization
visualize_augmentation(sample_image_path)

