import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from utils.utils import save_model

def train_model(model, train_loader, test_loader, device, criterion, optimizer, num_epochs=100):
    best_acc = 0.0
    early_stop = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_labels, all_preds = [], []

        with tqdm(train_loader, unit="batch") as tepoch:
            for images, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                tepoch.set_postfix(loss=running_loss / len(train_loader))

        train_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        train_acc = accuracy_score(all_labels, all_preds)
        train_loss = running_loss / len(train_loader)

        val_f1, val_acc, val_loss = evaluate(model, test_loader, device, criterion)

        print(f"Epoch: {epoch + 1} | train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, "
              f"train_f1: {train_f1:.4f}, val_f1: {val_f1:.4f}, val_acc: {val_acc:.4f}")

        if val_f1 > best_acc:
            best_acc = val_f1
            save_model(model, 'checkpoints/best_model.pth')
            print('Best model saved!')
            early_stop = 0
        else:
            early_stop += 1

        if early_stop > 20:
            break

def evaluate(model, dataloader, device, criterion):
    model.eval()
    val_labels, val_preds = [], []
    val_running_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(predicted.cpu().numpy())

    val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
    val_acc = accuracy_score(val_labels, val_preds)
    val_loss = val_running_loss / len(dataloader)
    return val_f1, val_acc, val_loss
