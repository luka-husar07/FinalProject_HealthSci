import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),  # reduce rotation if you want
    # Comment out or remove ColorJitter
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.0, hue=0.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
