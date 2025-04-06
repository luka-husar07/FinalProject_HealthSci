# %%
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from data_loader import dataset_load
from cnn_model import ultrasound_CNN
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ultrasound_CNN().to(device)
lr = 0.0005
num_epochs = 17


criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=lr)

best_val_acc = 0
best_model_state = None
train_losses, val_losses = [], []

train_loader, val_loader, _ = dataset_load(print_info=True)

train_accuracies = []
val_accuracies = []

#Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        labels = labels.view(-1)

        opt.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        opt.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100*correct/total
    avg_loss = train_loss/len(train_loader)

    train_accuracies.append(train_acc)

    #Validation Loop
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    model.eval()

    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_labels = val_labels.view(-1)
            val_outputs = model(val_images)
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()
            val_loss += criterion(val_outputs, val_labels).item()

    
    val_acc = 100*(val_correct/val_total)
    val_loss /= len(val_loader)
    #Early stopping if neccessary
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict()

    train_losses.append(avg_loss)
    val_losses.append(val_loss)

    val_accuracies.append(val_acc)

    
    #Logging metrics for tensorboard visualization
    '''
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    '''
 
    
    print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {avg_loss:.4f}  Val Loss: {val_loss:.4f}  Train Acc: {train_acc:.2f}%  Val Acc: {val_acc:.2f}%")

if best_model_state:
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), "best_cnn.pth")


# Plot loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png")  # Save the plot
#plt.show()  # Or comment this out if running on headless server

#plot accuracies
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_plot.png")

# Only show the plot if running interactively
if __name__ == "__main__":
    plt.show()

# %%
