import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from data.dataloader import CustomImageDataset, default_transform
from models.imageencoder import ImageEncoder
from trainer.train import train_model
import config

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = CustomImageDataset(config.train_csv, transform=default_transform)
    test_dataset = CustomImageDataset(config.test_csv, transform=default_transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    model = ImageEncoder(config.pth_file, config.num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_model(model, train_loader, test_loader, device, criterion, optimizer, num_epochs=config.num_epochs)

if __name__ == "__main__":
    main()
