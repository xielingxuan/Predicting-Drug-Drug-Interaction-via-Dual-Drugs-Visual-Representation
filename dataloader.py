import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PairedImagesDataset(Dataset):
    def __init__(self, root_A, root_B, transform=None):
        self.root_A = root_A
        self.root_B = root_B
        self.transform = transform

        self.images_A = os.listdir(root_A)
        self.images_B = os.listdir(root_B)
        self.num_images = min(len(self.images_A), len(self.images_B))

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        img_name_A = os.path.join(self.root_A, self.images_A[idx])
        img_name_B = os.path.join(self.root_B, self.images_B[idx])

        image_A = Image.open(img_name_A).convert('RGB')
        image_B = Image.open(img_name_B).convert('RGB')

        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)

        return image_A, image_B

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


dataset = PairedImagesDataset(root_A='testc', root_B='testd', transform=transform)


dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

