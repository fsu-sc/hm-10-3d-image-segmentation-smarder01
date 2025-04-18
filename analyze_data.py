import os
import numpy as np
import torch
import SimpleITK as sitk
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

# custom dataset class
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
    
# load dataset and print stats
dataset = HeartMRIDataset(mode = "train")
print(f"Number of training images: {len(dataset)}")

# Manually load first image with SimpleITK for spacing info
import SimpleITK as sitk
import os

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