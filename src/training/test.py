from src.models.cnn import CNN
from src.data.dataset import get_dataloaders
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config.config import load_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = load_config()
batch_size = config["training"]["batch_size"]
threshold = config["model"]["threshold"]
figure_dir = config["paths"]["figure_dir"]
os.makedirs(figure_dir, exist_ok=True)

def save_roc_curve(labels, probs, save_path):

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")

    plt.plot([0,1], [0,1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend()

    plt.savefig(save_path)
    plt.close()


def save_confusion_matrix(labels, preds, save_path):

    cm = confusion_matrix(labels, preds)

    plt.figure()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    plt.savefig(save_path)
    plt.close()    
    
    
def test():

    model = CNN().to(device)

    model.load_state_dict(torch.load("outputs/models/best_model.pth", map_location=device))

    _, _, test_loader = get_dataloaders(batch_size=batch_size)

    model.eval()

    n = 0
    acc = 0

    labels = []
    preds = []
    probs = []

    with torch.no_grad():

        for x, t in test_loader:
            n += t.size()[0]
            
            x = x.to(device)
            t = t.to(device).float()
            
            prob = model.forward(x)

            pred = torch.where(prob < threshold, 0, 1)
            acc += (pred == t).float().sum().item()
            labels.extend(t.detach().cpu().numpy())
            preds.extend(pred.detach().cpu().numpy())
            probs.extend(prob.detach().cpu().numpy())


    labels = np.array(labels)
    preds = np.array(preds)
    probs = np.array(probs)
    auc = roc_auc_score(labels, probs) 

    print("Test Accuracy:", acc / n)
    print("Test AUC:", auc)
    
    save_roc_curve(
    labels,
    probs,
    os.path.join(figure_dir, "roc_curve.png")
    )
    
    save_confusion_matrix(
    labels,
    preds,
    os.path.join(figure_dir, "confusion_matrix.png")
    )


if __name__ == "__main__":
    test()