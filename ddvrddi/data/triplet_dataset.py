import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import random

class TripletImageDataset(Dataset):
    def __init__(self, csv_files, transform=None, max_rows=None):
        self.data = pd.concat([pd.read_csv(f, nrows=max_rows) for f in csv_files], ignore_index=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor_path = self.data.iloc[idx, 0]
        positive_path = self.data.iloc[idx, 1]
        negative_idx = random.choice([i for i in range(len(self.data)) if i != idx])
        negative_path = self.data.iloc[negative_idx, 1]

        anchor = Image.open(anchor_path).convert("RGB")
        positive = Image.open(positive_path).convert("RGB")
        negative = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative
