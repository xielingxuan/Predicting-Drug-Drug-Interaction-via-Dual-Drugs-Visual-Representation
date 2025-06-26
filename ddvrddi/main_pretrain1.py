import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from models.imageencoder import ImageEncoder
from data.triplet_dataset import TripletImageDataset
from trainer.pretrain_triplet import pretrain_triplet
import config_pretrain1 as config

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = TripletImageDataset(config.csv_files, transform=transform, max_rows=config.max_rows)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                            num_workers=config.num_workers, pin_memory=True)

    model = ImageEncoder(pth_file=config.pretrained_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.TripletMarginLoss(margin=1.0)

    pretrain_triplet(model, dataloader, optimizer, criterion, device,
                     config.num_epochs, config.output_model_path)

if __name__ == '__main__':
    main()
