# %%
import torch
from data_loader import dataset_load
from cnn_model import ultrasound_CNN
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def test_model(model_path="best_cnn.pth", batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test set only
    _, _, test_loader = dataset_load(batch_size=batch_size)

    # Initialize model and load weights
    model = ultrasound_CNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.view(-1)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = 100 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}% ({correct}/{total})")

    # Define class labels (modify if your class order is different)
    class_names = ["normal", "benign", "malignant"]

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    # plt.show()  # Uncomment if running in a GUI



    return all_preds, all_labels


if __name__ == "__main__":
    preds, labels = test_model()

# %%
