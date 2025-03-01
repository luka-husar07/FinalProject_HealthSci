import data_loader

import matplotlib.pyplot as plt

class_names = ["Normal", "Benign", "Malignant"]

# Get a batch of images
images, labels = next(iter(data_loader.dataloader))

# Convert images to numpy format for visualization
images = images.permute(0, 2, 3, 1).numpy()  # Change (B, C, H, W) -> (B, H, W, C)

# Plot some images
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(images[i])
    ax.set_title(class_names[labels[i].item()])
    ax.axis("off")

plt.show()
