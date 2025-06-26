import os
from PIL import Image
from random import shuffle
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

class JigsawPuzzleDataset(Dataset):
    def __init__(self, image_dirs, transform=None, rows=3, cols=6):
        self.image_files = []
        for image_dir in image_dirs:
            self.image_files += [
                os.path.join(image_dir, f)
                for f in os.listdir(image_dir)
                if os.path.isfile(os.path.join(image_dir, f))
            ]
        self.transform = transform
        self.rows = rows
        self.cols = cols
        self.num_tiles = rows * cols

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        tiles, perm = self._create_jigsaw(image)
        return tiles, perm

    def _create_jigsaw(self, image):
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        w, h = image.size
        tile_w, tile_h = w // self.cols, h // self.rows

        tiles = []
        for i in range(self.rows):
            for j in range(self.cols):
                tile = image.crop((j * tile_w, i * tile_h, (j + 1) * tile_w, (i + 1) * tile_h))
                if self.transform:
                    tile = self.transform(tile)
                tiles.append(tile)

        perm = list(range(self.num_tiles))
        shuffle(perm)
        tiles = [tiles[i] for i in perm]

        return torch.stack(tiles), torch.tensor(perm)
