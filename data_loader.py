import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from augmentation import get_augmented_transform

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Light training transform (can add more later)
train_transform = get_augmented_transform()

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

        image = Image.open(img_path).convert("L")  # Convert to 3-channel image

        if self.transform:
            image = self.transform(image)

        return image, label

# Define Data Transformations
transform_default = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 for CNN models
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize between -1 and 1
])

# Download dataset using kagglehub
#path = kagglehub.dataset_download("aryashah2k/breast-ultrasound-images-dataset")
#print("Path to dataset files:", path)


def dataset_load(path = "/research/osz09/shared_students/Spring_2025/team1/dataset", 
                 print_info=False, 
                 transform=transform_default,
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                 batch_size=16):
    # Set the dataset path
    dataset_path = os.path.join(path, "Dataset_BUSI_with_GT")

    # Initialize dataset
    full_dataset = BreastUltrasoundDataset(root_dir=dataset_path, transform=transform)


    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])

    train_set.dataset.transform = train_transform
    val_set.dataset.transform = val_test_transform
    test_set.dataset.transform = val_test_transform

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # Example: Fetch a batch of images and labels
    if print_info:
        print(f"Train: {len(train_set)} samples")
        print(f"Validation: {len(val_set)} samples")
        print(f"Test: {len(test_set)} samples")

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    dataset_load(print_info=True)
