csv_files = [f'testrotate{i}.csv' for i in range(1, 15)]

pretrained_weights = 'myimagemol.pth'   # 或设为 None
output_model_path = 'checkpoints/imageencoder_encoder47.pth'

max_rows = None
batch_size = 64
learning_rate = 1e-6
num_epochs = 5
num_workers = 8
