import torch
import clip
from PIL import Image

# 加载 CLIP 模型
model, preprocess = clip.load("ViT-B/32", device="cpu")
image_encoder = model.visual

class YourImageEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model

    def forward(self, x):
        return self.encoder(x)

your_image_encoder = YourImageEncoder(image_encoder)

# 加载权重文件，并映射键名称
state_dict = torch.load('image_encoder_finetuned.pth', map_location="cpu")
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace("encoder.", "")  # 修改键名以匹配自定义编码器
    new_state_dict[new_key] = value

# 加载映射后的权重到自定义编码器
your_image_encoder.load_state_dict(new_state_dict, strict=False)

# 从该路径上加载要提取特征的图片
image_path = 'output_image.png'
#这块应该再输出另一张图片的特征，然后比较两个特征的相似度
input_image = Image.open(image_path)

# 对图像进行预处理
input_tensor = preprocess(input_image).unsqueeze(0)

# 使用加载后的编码器提取特征
with torch.no_grad():
    image_features = your_image_encoder(input_tensor)

print("图片特征形状:", image_features)