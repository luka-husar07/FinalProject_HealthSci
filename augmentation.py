import torchvision.transforms as transforms

def get_augmented_transform():
    """Returns a transformation pipeline for data augmentation."""
    return transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.RandomRotation(degrees=30),  
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  
        transforms.RandomResizedCrop(size=224, scale=(0.9, 1.1)),  
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        transforms.RandomVerticalFlip(p=0.5),  
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5], std=[0.5])  
    ])