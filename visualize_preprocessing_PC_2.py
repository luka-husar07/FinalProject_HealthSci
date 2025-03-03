"""
visualize_preprocessing.py
--------------------------------
This script demonstrates how the preprocessing pipeline in preprocessing.py
transforms a sample breast ultrasound image. It displays side-by-side
comparison of the original vs. the preprocessed image.
"""

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import os

# Import the custom transform from preprocessing.py
from preprocessing import transform

# Dataset path: Update if it differs on your system!
DATASET_PATH = "C:/Users/piyus/.cache/kagglehub/datasets/aryashah2k/breast-ultrasound-images-dataset/versions/1/Dataset_BUSI_with_GT"

# Ensure the dataset path exists
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")

# List available categories
categories = os.listdir(DATASET_PATH)
print("Available categories:", categories)

# Choose one category: "normal", "benign", or "malignant"
chosen_category = "benign"  # Change if needed
category_path = os.path.join(DATASET_PATH, chosen_category)

if not os.path.exists(category_path):
    raise FileNotFoundError(f"Category path not found: {category_path}")

# Gather images in this category (PNG files)
images = [f for f in os.listdir(category_path) if f.lower().endswith(".png")]
if len(images) == 0:
    raise FileNotFoundError(f"No images found in {category_path}")

# Pick any image from this list
sample_image_path = os.path.join(category_path, images[0])
print("Using sample image:", sample_image_path)

def load_sample_image(image_path):
    """
    Opens an image file, ensures it's in RGB format (3 channels).
    """
    image = Image.open(image_path).convert("RGB")
    return image

def visualize_preprocessing(image_path):
    """
    Applies the preprocessing pipeline to an image and displays
    the Original vs. Preprocessed result side by side.
    """
    # Load original image
    original_image = load_sample_image(image_path)

    # Apply our transform (Resize, Flip, Rotate, ColorJitter, Normalize)
    transformed_tensor = transform(original_image)

    # Convert transformed tensor back to PIL for visualization
    transformed_image = transforms.ToPILImage()(transformed_tensor)

    # Plot them side by side
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    
    # Left: Original Image
    ax[0].imshow(original_image, cmap="gray")
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Right: Transformed / Preprocessed
    ax[1].imshow(transformed_image, cmap="gray")
    ax[1].set_title("Preprocessed Image")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()

# Execute visualization
visualize_preprocessing(sample_image_path)
