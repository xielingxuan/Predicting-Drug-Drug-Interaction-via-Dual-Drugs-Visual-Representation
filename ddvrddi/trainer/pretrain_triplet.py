import torch
from tqdm import tqdm

def pretrain_triplet(model, dataloader, optimizer, criterion, device, num_epochs, save_path):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for img1, img2, img3 in loop:
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            img3 = img3.to(device, non_blocking=True)

            optimizer.zero_grad()
            embed1 = model(img1)
            embed2 = model(img2)
            embed3 = model(img3)
            loss = criterion(embed1, embed2, embed3)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix({'Loss': loss.item()})

        print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {epoch_loss / len(dataloader):.4f}")

    torch.save(model.img_encoder.state_dict(), save_path)
    print(f"模型已保存到 {save_path}")
