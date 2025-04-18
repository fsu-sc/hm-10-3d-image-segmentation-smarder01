import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from analyze_data import HeartMRIDataset
from mymodel import UNet3D

# dice loss function
def dice_loss(pred, target, smooth = 1e-5):
    # pred: raw logits from model [B, 1, D, H, W]
    # target: ground truth mask [B, 1, D, H, W]
    pred = torch.sigmoid(pred)
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(1)
    union = pred.sum(dim = 1) + target.sum(dim = 1)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    loss = 1 - dice
    return loss.mean()

# training function
def train_model(epochs=50, batch_size=2, learning_rate=1e-4, log_dir="runs/heart_seg", checkpoint_dir="checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # datsets and loaders
    train_dataset = HeartMRIDataset(mode="train")
    val_dataset = HeartMRIDataset(mode="val")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # model and optimizer
    model = UNet3D(in_channels = 1, out_channels = 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    # tensorboard and checkpoints
    writer = SummaryWriter(log_dir = log_dir)
    os.makedirs(checkpoint_dir, exist_ok = True)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for images, masks, _, _ in train_loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            outputs = model(images)
            loss = dice_loss(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)

        # validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, masks, _, _ in val_loader:
                images = images.to(device, dtype=torch.float32)
                masks = masks.to(device, dtype=torch.float32)

                outputs = model(images)
                loss = dice_loss(outputs, masks)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)

        print(f"[Epoch {epoch+1:02d}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print("New best model saved.")

    # Final save
    final_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    writer.close()
    print("🏁 Training complete.")

    return model