import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image


output_dir = 'Ryu_training'
os.makedirs(output_dir, exist_ok=True)

# 读取CSV文件
csv_file = 'RYU-train.csv'  # 替换为你的CSV文件路径
df = pd.read_csv(csv_file)

# 创建一个空的DataFrame用于存储结果
result_data = []

# 遍历每一行，读取SMILES并生成图像
for idx, row in df.iterrows():
    smiles1 = row.iloc[0]
    id_value = row.iloc[1]
    smiles2 = row.iloc[2]

    # 检查第一列和第三列是否为空
    if pd.isna(smiles1) or pd.isna(smiles2):
        print(f'Skipping row {idx + 1} due to missing SMILES')
        continue

    molecule1 = Chem.MolFromSmiles(smiles1)
    molecule2 = Chem.MolFromSmiles(smiles2)

    if molecule1 is not None and molecule2 is not None:
        # 生成两个分子的图像
        img1 = Draw.MolToImage(molecule1, size=(224, 224))
        img2 = Draw.MolToImage(molecule2, size=(224, 224))

        # 拼接图像
        combined_img = Image.new('RGB', (448, 224))  # 创建一个新的空白图像，宽度是两个图像的总和
        combined_img.paste(img1, (0, 0))
        combined_img.paste(img2, (224, 0))

        # 保存拼接后的图像
        img_path = os.path.join(output_dir, f'combined_molecule_{idx + 1}.png')
        combined_img.save(img_path)

        # 将结果存储到result_data
        result_data.append({'id': id_value, 'image_path': img_path})
        print(f'Saved {img_path}')
    else:
        print(f'Invalid SMILES at row {idx + 1}')

# 将结果数据转为DataFrame并保存到CSV文件
result_df = pd.DataFrame(result_data)
result_df.to_csv('Ryu_training.csv', index=False)
print("All images have been processed and saved.")