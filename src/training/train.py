from src.models.cnn import CNN
from src.data.dataset import get_dataloaders
import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.metrics import roc_auc_score
from config.config import load_config

# config
config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = config["training"]["batch_size"]
lr = config["training"]["lr"]
n_epochs = config["training"]["epochs"]
threshold = config["model"]["threshold"]
save_dir = config["paths"]["model_dir"]
os.makedirs(save_dir, exist_ok=True)


def train(model, train_loader, optimizer, criterion):
    model.train()
    n_train = 0
    acc_train = 0
    losses_train = []
    labels_tra = []
    preds_tra = []
    
    for x, t in train_loader:
        n_train += t.size()[0]

        model.zero_grad()

        x = x.to(device)
        t = t.to(device).float()

        prob = model.forward(x)

        loss = criterion(prob, t)

        loss.backward()

        optimizer.step() 

        pred = torch.where(prob < threshold, 0, 1)

        acc_train += (pred == t).float().sum().item()
        losses_train.append(loss.item())
        
        labels_tra.extend(t.detach().cpu().numpy())
        preds_tra.extend(pred.detach().cpu().numpy())
    
    labels_tra = np.array(labels_tra)
    preds_tra = np.array(preds_tra)
    auc_tra = roc_auc_score(labels_tra, preds_tra)
    
    return np.mean(losses_train), acc_train / n_train, auc_tra


def evaluate(model, val_loader, criterion):
    model.eval()
    n_val = 0
    acc_val = 0
    losses_valid = []
    labels_val = []
    preds_val = []
    
    with torch.no_grad():
    
        for x, t in val_loader:
            n_val += t.size()[0]

            x = x.to(device)
            t = t.to(device).float()

            prob = model.forward(x)

            loss = criterion(prob, t)

            pred = torch.where(prob < threshold, 0, 1)

            acc_val += (pred == t).float().sum().item()
            losses_valid.append(loss.item())
            
            labels_val.extend(t.detach().cpu().numpy())
            preds_val.extend(pred.detach().cpu().numpy())
        
    labels_val = np.array(labels_val)
    preds_val = np.array(preds_val)
    auc_val = roc_auc_score(labels_val, preds_val)
    
    return np.mean(losses_valid), acc_val / n_val, auc_val


def main():

    model = CNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.BCELoss()

    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)

    best_auc = 0

    for epoch in range(n_epochs):

        train_loss, train_acc, train_auc = train(
            model, train_loader, optimizer, criterion
        )

        val_loss, val_acc, val_auc = evaluate(
            model, val_loader, criterion
        )

        print(
            "EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}, AUC: {:.3f}], "
            "Valid [Loss: {:.3f}, Accuracy: {:.3f}, AUC: {:.3f}]".format(
                epoch,
                train_loss,
                train_acc,
                train_auc,
                val_loss,
                val_acc,
                val_auc,
            )
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print("Best model saved with AUC: {:.3f}".format(best_auc))

    print("Training Finished.")


if __name__ == "__main__":
    main()