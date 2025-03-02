import data_loader

import matplotlib.pyplot as plt

class_names = ["Normal", "Benign", "Malignant"]

def image_visualization(dataloader):
    class_names = ["Normal", "Benign", "Malignant"]

    # Loop through the entire dataloader
    for images, labels in dataloader:
        # Convert images to numpy format for visualization
        images = images.permute(0, 2, 3, 1).numpy()  # Change (B, C, H, W) -> (B, H, W, C)

        # Plot images in the batch
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for i, ax in enumerate(axes.flatten()):
            if i >= len(images):
                break
            # Convert the image from tensor to numpy and denormalize
            img = images[i]
            ax.imshow(img)
            ax.set_title(class_names[labels[i].item()])
            ax.axis("off")

        plt.show()
        break  # Remove this break to visualize all batches

'''
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
'''