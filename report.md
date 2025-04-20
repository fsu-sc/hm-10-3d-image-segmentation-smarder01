# Homework Report

## Personal Details
**Name:** Sydney Marder
**Date:** 4/20/25
**Course:** ISC 5935
**Instructor:** Olmo S. Zavala-Romero

## Homework Questions and Answers

### Dataset Access and Loading
For this assignment, I used the pre-downloaded Heart MRI dataset from the Medical Segmentation Decathlon challenge, provided by the instructor. The dataset was accessed from the shared class directory:
```python
data_dir = "/home/osz09/DATA_SharedClasses/SharedDatasets/MedicalDecathlon/Task02_Heart"
```
The `HeartMRIDataset` class is a custom PyTorch `Dataset` implementation that automates the loading and preprocessing of the MRI volumes and segmentation masks. It takes care of:

- Parsing image and label file paths based on a training or validation split
- Loading NIfTI files using `SimpleITK`
- Normalizing the MRI intensities to a [0, 1] scale
- Applying a random crop of size `(64, 128, 128)` to both the image and label
- Returning the preprocessed image and mask as PyTorch tensors with shape `[1, D, H, W]`

By handling all of this inside the dataset class, the DataLoader can easily batch and serve clean, consistent 3D inputs during training and evaluation.


My code:
```python
class HeartMRIDataset(Dataset):
    def __init__(self, root="/home/osz09/DATA_SharedClasses/SharedDatasets/MedicalDecathlon/Task02_Heart", mode="train"):
        image_root = os.path.join(root, "imagesTr")
        label_root = os.path.join(root, "labelsTr")

        all_cases = sorted(os.listdir(image_root))
        cutoff = int(0.8 * len(all_cases))

        self.image_paths = []
        self.label_paths = []

        if mode == "train":
            selected_cases = all_cases[:cutoff]
        else:
            selected_cases = all_cases[cutoff:]

        for case in selected_cases:
            img_path = os.path.join(image_root, case)
            lbl_path = os.path.join(label_root, case.replace("_0000", ""))
            self.image_paths.append(img_path)
            self.label_paths.append(lbl_path)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load NIfTI image and label
        img_sitk = sitk.ReadImage(self.image_paths[idx])
        lbl_sitk = sitk.ReadImage(self.label_paths[idx])

        img_np = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
        lbl_np = sitk.GetArrayFromImage(lbl_sitk).astype(np.float32)

        # Normalize image to [0, 1]
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

        # Define crop size
        crop_d, crop_h, crop_w = (64, 128, 128)  # You can adjust these if needed
        D, H, W = img_np.shape

        # If the volume is smaller than crop size, raise an error
        if D < crop_d or H < crop_h or W < crop_w:
            raise ValueError(f"Volume too small for crop size: got ({D}, {H}, {W}), need at least ({crop_d}, {crop_h}, {crop_w})")

        # Random crop (you could use center crop instead if you prefer)
        start_d = np.random.randint(0, D - crop_d + 1)
        start_h = np.random.randint(0, H - crop_h + 1)
        start_w = np.random.randint(0, W - crop_w + 1)

        # Apply the same crop to both image and label
        img_np = img_np[start_d:start_d + crop_d, start_h:start_h + crop_h, start_w:start_w + crop_w]
        lbl_np = lbl_np[start_d:start_d + crop_d, start_h:start_h + crop_h, start_w:start_w + crop_w]

        # Convert to torch tensors and add channel dimension: [C, D, H, W]
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)
        lbl_tensor = torch.from_numpy(lbl_np).unsqueeze(0)

        return img_tensor, lbl_tensor
```

---

### Data Exploration
After loading the dataset, I performed basic data exploration to better understand the structure and content of the MRI volumes. I inspected the number of training images, their voxel spacing, and their dimensionality using `SimpleITK`.

I also visualized sample slices from three standard anatomical planes: axial, sagittal, and coronal. This helped me verify the quality of both the MRI scans and their associated segmentation masks. I saved this visualization as `sample_slices.png`.

Finally, I calculated the distribution of segmentation volumes (i.e., number of positive voxels per mask). This histogram, saved as `volume_distribution.png`, helps confirm that the dataset includes a variety of anatomical cases with differing atrial sizes — which is important for training a generalizable model.

My code:
```python
# load dataset and print stats
dataset = HeartMRIDataset(mode = "train")
print(f"Number of training images: {len(dataset)}")

# Manually load first image with SimpleITK for spacing info
first_image_path = dataset.image_paths[0]
img_sitk = sitk.ReadImage(first_image_path)

spacing = img_sitk.GetSpacing()
size = img_sitk.GetSize()

print(f"Image dimensions (W, H, D): {size}")
print(f"Voxel spacing (x, y, z): {spacing}")

# visualize sample slices
def visualize_sample(dataset, index=0):
    img_tensor, lbl_tensor = dataset[index]
    img = img_tensor.squeeze(0).numpy()
    lbl = lbl_tensor.squeeze(0).numpy()

    mid_axial = img.shape[0] // 2
    mid_sagittal = img.shape[1] // 2
    mid_coronal = img.shape[2] // 2

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    # Axial
    axes[0, 0].imshow(img[mid_axial], cmap="gray")
    axes[0, 0].set_title("Axial MRI")
    axes[1, 0].imshow(lbl[mid_axial], cmap="Reds")
    axes[1, 0].set_title("Axial Mask")

    # Sagittal
    axes[0, 1].imshow(img[:, mid_sagittal, :], cmap="gray")
    axes[0, 1].set_title("Sagittal MRI")
    axes[1, 1].imshow(lbl[:, mid_sagittal, :], cmap="Reds")
    axes[1, 1].set_title("Sagittal Mask")

    # Coronal
    axes[0, 2].imshow(img[:, :, mid_coronal], cmap="gray")
    axes[0, 2].set_title("Coronal MRI")
    axes[1, 2].imshow(lbl[:, :, mid_coronal], cmap="Reds")
    axes[1, 2].set_title("Coronal Mask")

    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("sample_slices.png")
    plt.close()
    print("✅ Saved slice visualization as sample_slices.png")

visualize_sample(dataset, index=0)

#  distribution of segmentation volumes
def volume_distribution(dataset):
    volumes = []
    for i in range(len(dataset)):
        _, lbl_tensor = dataset[i]
        volumes.append(torch.sum(lbl_tensor > 0).item())

    plt.figure(figsize=(8, 5))
    plt.hist(volumes, bins=10, color="skyblue", edgecolor="black")
    plt.title("Distribution of Segmentation Volumes")
    plt.xlabel("Volume (voxels)")
    plt.ylabel("Number of Cases")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("volume_distribution.png")
    plt.close()
    print("✅ Saved volume histogram as volume_distribution.png")

volume_distribution(dataset)
```

**Basic Statistics**
- Number of training images: 16
- Image dimensions (W, H, D): (320, 320, 130)
- Voxel spacing (x, y, z): (1.25, 1.25, 1.3700000047683716)-

**Sample Slices**

**Distribution of segmentation volumes**

---

### Model Architecture
For this task, I implemented a 3D U-Net-style convolutional neural network in PyTorch. The U-Net is well-suited for medical image segmentation because it combines a contracting encoder path (which captures context) with an expanding decoder path (which enables precise localization), along with skip connections that preserve spatial detail.

The model consists of:

- An **encoder** with three blocks of 3D convolutions followed by max pooling, progressively reducing the spatial resolution while increasing feature depth.
- A **bottleneck** layer that captures high-level features.
- A **decoder** with three upsampling blocks using transposed convolutions followed by 3D convolutions, restoring spatial resolution.
- **Skip connections** between each encoder and decoder level, which help recover fine-grained details lost during downsampling.
- A final `Conv3D` layer with kernel size 1 that outputs a 1-channel segmentation map.

Each convolution block is implemented using a helper class called `DoubleConv3D`, which includes two `Conv3D → BatchNorm3D → ReLU` sequences.

The total number of trainable parameters in the model is 5,602,529.

My code:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# define a reusable double 3D convolution block
class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    
# define the U-Net model
class UNet3D(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1):
        super().__init__()

        # Encoder (Downsampling)
        self.enc1 = DoubleConv3D(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv3D(32, 64)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = DoubleConv3D(64, 128)
        self.pool3 = nn.MaxPool3d(2)

        # bottleneck
        self.bottleneck = DoubleConv3D(128, 256)

        # decoder (Upsampling)
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv3D(256, 128)

        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3D(128, 64)

        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(64, 32)

        # final output
        self.final_conv = nn.Conv3d(32, out_channels, kernel_size = 1)

    def forward(self, x):
        # encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))

        #bottleneck
        x4 = self.bottleneck(self.pool3(x3))

        # decoder
        d3 = self.up3(x4)
        d3 = torch.cat([d3, x3], dim = 1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec1(d1)

        # output segmentation map
        out = self.final_conv(d1)
        return out
```

**screenshot of model architecture**

---

### Training Implementation
To train the 3D U-Net model, I implemented a training loop using PyTorch. The training setup includes:

- **Loss function:** I used a custom implementation of Dice loss, which is well-suited for segmentation tasks. Dice loss directly measures overlap between the predicted and ground truth masks and works well with class imbalance.
- **Optimizer:** Adam optimizer with a learning rate of `1e-4` was used to update model weights.
- **Training loop:** For each epoch, the model processes all batches in the training set, computes the loss, performs backpropagation, and updates the weights.
- **Validation loop:** After each training epoch, the model is evaluated on the validation set, and the average Dice loss is logged.
- **Checkpointing:** The model with the best validation loss is saved as `best_model.pth`. A final model is also saved at the end of training.
- **Batching:** A batch size of 2 was used to balance memory usage and gradient stability.

To monitor progress, I logged metrics to TensorBoard using `SummaryWriter`. Specifically, I recorded:
- **Training Dice loss curve**
- **Validation Dice loss curve**
- **Model architecture graph** (logged once at the beginning using a real sample input)

My code:
```python
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

    # Save model architecture to TensorBoard using one batch
    sample_input = next(iter(train_loader))[0].to(device)
    writer.add_graph(model, sample_input)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for images, masks in train_loader:
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
            for images, masks in val_loader:
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
    print("Training complete.")

    return model
```

**Training Dice Loss Curve (TensorBoard)**

**Validation Dice Loss Curve (TensorBoard)**

---

### Model Evaluation
After training, I evaluated the model’s performance on the validation set by visualizing predictions and calculating Dice scores. I used the best saved checkpoint, `best_model.pth` (located in the checkpoints folder), loaded it into evaluation mode, and ran it on unseen validation samples.

For qualitative evaluation, I visualized the model’s segmentation predictions on a mid-slice of a 3D MRI volume, alongside the corresponding input scan and ground truth mask. This helps assess how accurately the model identifies the left atrium.

The prediction output was passed through a sigmoid activation and thresholded at 0.5 to create a binary mask. Below is the visualization:

📷 *Sample prediction vs. ground truth:*

![Prediction Example](path/to/prediction_sample.png)

To quantify segmentation performance, I used Dice similarity score — a metric that measures overlap between the predicted and ground truth masks. Higher Dice values (closer to 1.0) indicate better performance.

My code:
```python
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
        image, mask = val_dataset[index]
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
```

**screenshot of sample prediction**

---