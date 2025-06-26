image_dirs = ['traindata1', 'traindata2', 'traindata3', 'traindata4', 'traindata5']

transform = transforms.Compose([
    transforms.Resize((448, 224)),
    transforms.ToTensor(),
])

pretrained_weights = 'myimagemol.pth'
batch_size = 16
num_epochs = 50
learning_rate = 0.001
num_workers = 4
