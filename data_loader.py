import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import kagglehub

#Set the default path to the dataset
#path = "/research/osz09/shared_students/Spring_2025"

class BreastUltrasoundDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory containing the dataset folders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ["normal", "benign", "malignant"]  # Folder names
        self.image_paths = []
        self.labels = []

        # Load all images (excluding mask files)
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if "_mask" not in img_name:  # Exclude masks
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")  # Convert to 3-channel image

        if self.transform:
            image = self.transform(image)

        return image, label

# Define Data Transformations
transform_default = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 for CNN models
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0], std=[1])  # Normalize between -1 and 1
])

# Download dataset using kagglehub
#path = kagglehub.dataset_download("aryashah2k/breast-ultrasound-images-dataset")
#print("Path to dataset files:", path)
def dataset_load(path = "/research/osz09/shared_students/Spring_2025", print_info=False, transform=transform_default):
    # Set the dataset path
    dataset_path = os.path.join(path, "Dataset_BUSI_with_GT")

    # Initialize dataset
    breast_ultrasound_dataset = BreastUltrasoundDataset(root_dir=dataset_path, transform=transform)

    # Create DataLoader
    dataloader = DataLoader(breast_ultrasound_dataset, batch_size=16, shuffle=True, num_workers=2)

    # Example: Fetch a batch of images and labels
    if print_info:
        for images, labels in dataloader:
            print(f"Batch Image Shape: {images.shape}")  # Expected: (batch_size, 3, 224, 224)
            print(f"Batch Labels: {labels}")
            break  # Only show first batch
    return dataloader

if __name__ == "__main__":
    dataset_load(kagglehub.dataset_download("aryashah2k/breast-ultrasound-images-dataset"), print_info=True)
