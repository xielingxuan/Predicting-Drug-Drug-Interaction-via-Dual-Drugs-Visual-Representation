import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.imageencoder import ImageEncoder
from data.jigsaw_dataset import JigsawPuzzleDataset
from trainer.pretrain_jigsaw import train_jigsaw
import config_pretrain2 as config

class JigsawModel(nn.Module):
    def __init__(self, img_encoder, embed_dim=50176, num_tiles=18):
        super(JigsawModel, self).__init__()
        self.img_encoder = img_encoder
        self.num_tiles = num_tiles
        self.fc = nn.Linear(embed_dim, num_tiles)

    def forward(self, x):
        x = x.view(-1, 3, x.size(3), x.size(4))
        features = self.img_encoder(x)
        features = features.view(features.size(0), -1)
        output = self.fc(features)
        return output

def main():
    dataset = JigsawPuzzleDataset(
        image_dirs=config.image_dirs,
        transform=config.transform,
        rows=3, cols=6
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                            num_workers=config.num_workers)

    encoder = ImageEncoder(pth_file=config.pretrained_weights).img_encoder
    model = JigsawModel(encoder, embed_dim=50176, num_tiles=18)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    train_jigsaw(model, dataloader, criterion, optimizer,
                 num_epochs=config.num_epochs, device='cuda')

if __name__ == "__main__":
    main()
