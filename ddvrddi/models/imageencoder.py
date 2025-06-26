import torch
import torch.nn as nn
import torchvision

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.residual_connection = nn.Sequential()
        if in_channels != out_channels:
            self.residual_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.residual_connection(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class ImageEncoder(nn.Module):
    def __init__(self, pth_file, num_classes):
        super(ImageEncoder, self).__init__()
        self.img_encoder = torchvision.models.resnet18(weights=None)
        self.img_encoder = nn.Sequential(*list(self.img_encoder.children())[:-2])
        self._load_weights(pth_file)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.residual_block = ResidualBlock(512, 512)
        self.classifier = nn.Linear(512, num_classes)

    def _load_weights(self, pth_file):
        state_dict = torch.load(pth_file)
        model_state_dict = self.img_encoder.state_dict()
        new_state_dict = {
            k: v for (k, v), (mk, mv) in zip(state_dict.items(), model_state_dict.items())
            if mv.shape == v.shape
        }
        self.img_encoder.load_state_dict(new_state_dict, strict=False)

    def forward(self, x):
        x = self.img_encoder(x)
        x = self.global_avg_pool(x)
        x = self.residual_block(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
