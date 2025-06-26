import torch
from tqdm import tqdm

def train_jigsaw(model, dataloader, criterion, optimizer, num_epochs=50, device='cuda'):
    model.train()
    model.to(device)
    for epoch in range(num_epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for tiles, perm in progress_bar:
            tiles, perm = tiles.to(device), perm.to(device)
            perm = perm.view(-1)
            optimizer.zero_grad()
            output = model(tiles)
            output = output.view(-1, model.num_tiles)
            loss = criterion(output, perm)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / len(dataloader))
            torch.cuda.empty_cache()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}')

    torch.save(model.img_encoder.state_dict(), 'checkpoints/jigsaw_imageencoder_v2.pth')
