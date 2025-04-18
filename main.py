import torch
import matplotlib.pyplot as plt
import os

from training import train_model, dice_loss
from analyze_data import HeartMRIDataset
from mymodel import UNet3D

# train the model
print("Training the model...")
model = train_model(epochs = 50, batch_size = 2, learning_rate = 1e-4)

# load best model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
model.to(device)
model.eval()
print("Loaded best model checkpoint.")

# visualize predictions on val set
val_dataset = HeartMRIDataset(mode = "val")


def visualize_prediction(index=0):
    with torch.no_grad():
        image, mask, _, _ = val_dataset[index]
        image = image.unsqueeze(0).to(device, dtype=torch.float32)  # [1, 1, D, H, W]
        mask = mask.squeeze().cpu().numpy()

        output = model(image)
        pred = torch.sigmoid(output).squeeze().cpu().numpy()

        # Use middle axial slice
        mid_slice = pred.shape[0] // 2

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(image.squeeze().cpu().numpy()[mid_slice], cmap="gray")
        axs[0].set_title("Input MRI")

        axs[1].imshow(mask[mid_slice], cmap="Reds")
        axs[1].set_title("Ground Truth")

        axs[2].imshow(pred[mid_slice] > 0.5, cmap="Blues")
        axs[2].set_title("Predicted Mask")

        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig("prediction_sample.png")
        plt.close()
        print("🖼️ Saved prediction to prediction_sample.png")

visualize_prediction(index=0)

