import torch
import clip
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 CLIP 模型
model, preprocess = clip.load("ViT-B/32", device=device)
image_encoder = model.visual
text_encoder = model.transformer

for param in image_encoder.parameters():
    param.data = param.data.to(torch.float)
# 控制数据格式一致

# 冻结 text encoder 的参数
for param in text_encoder.parameters():
    param.requires_grad = False

# 数据加载
from dataloader import PairedImagesDataset

# 定义损失函数，逻辑就是最小化两个图片之间的loss
def pairwise_loss(image_features_A, image_features_B):
    return F.mse_loss(image_features_A, image_features_B)

# 定义优化器
optimizer = optim.Adam(image_encoder.parameters(), lr=0.001)

# 图像预处理的代码
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = PairedImagesDataset(root_A='testc', root_B='testd', transform=transform)
#testc存的是处理过后的数据（旋转处理等）testd是未经处理直接拼接的数据
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 开始微调训练
num_epochs = 5
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_images_A, batch_images_B in dataloader:
        optimizer.zero_grad()
        batch_images_A = batch_images_A.to(device)
        batch_images_B = batch_images_B.to(device)

        batch_images_A = batch_images_A.to(torch.float)
        image_features_A = image_encoder(batch_images_A)
        image_features_B = image_encoder(batch_images_B)

        loss = pairwise_loss(image_features_A, image_features_B)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {total_loss / len(dataloader)}')

# 保存训练后的 Image Encoder 权重
# torch.save(image_encoder.state_dict(), 'image_encoder_finetuned.pth')
print(" 权重已保存")