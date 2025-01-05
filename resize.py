import nibabel as nib
import random
import os
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torch
from pathlib import Path
from scipy.ndimage import zoom
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def resize_image(img_data):
    # Define target shape for the resized image
    target_shape = (256, 256, 180)
    
    # Convert input to a PyTorch tensor and add batch and channel dimensions
    img_tensor = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, D, H, W)

    # Resize using torch.nn.functional.interpolate (trilinear interpolation)
    resized_tensor = F.interpolate(img_tensor, size=target_shape, mode='trilinear', align_corners=False)

    resized_data = resized_tensor.squeeze() # Tensor Shape: (256, 256, 180)
    return resized_data


class NRandomCrop:
    def __init__(self, crop_size=(96, 96, 64), crop_no=10):
        self.crop_size = crop_size
        self.crop_no = crop_no

    def __call__(self, images, labels):
        crops = []
        label_crops = []
        relabeled_labels = []
        d, h, w = images.shape
        pos_counter, neg_counter = 0, 0

        while len(crops) < self.crop_no:
            d_start = random.randint(0, d - self.crop_size[0])
            h_start = random.randint(0, h - self.crop_size[1])
            w_start = random.randint(0, w - self.crop_size[2])

            crop = images[d_start:d_start + self.crop_size[0],
                          h_start:h_start + self.crop_size[1],
                          w_start:w_start + self.crop_size[2]]
            label_crop = labels[d_start:d_start + self.crop_size[0],
                                h_start:h_start + self.crop_size[1],
                                w_start:w_start + self.crop_size[2]]

            max_label = torch.max(torch.tensor(label_crop.flatten()), dim=0)[0].item()
            relabeled = 1 if max_label == 2 else 0

            if relabeled == 1 and pos_counter < self.crop_no / 2:
                crops.append(crop)
                label_crops.append(label_crop)
                relabeled_labels.append(relabeled)
                pos_counter += 1
            elif relabeled == 0 and neg_counter < self.crop_no / 2:
                crops.append(crop)
                label_crops.append(label_crop)
                relabeled_labels.append(relabeled)
                neg_counter += 1

        resized_crops = [resize_image(crop) for crop in crops]

        print(f"Resized crops: {len(resized_crops)}")
        print(f"Relabeled labels: {len(relabeled_labels)}")

        return resized_crops, relabeled_labels


class NIfTIDataset(Dataset):
    def __init__(self, dir_path, transform=None):
        """
        Custom Dataset for loading NIfTI images from a directory.
        Args:
            dir_path (str): Path to the folder containing the images and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dir_path = dir_path
        self.files = []
        self.transform = transform
        self._list_files_in_dir()

    def _list_files_in_dir(self):
        self.training_files = []
        self.label_files = []

        for dirpath, dirnames, filenames in os.walk(self.dir_path):
            for filename in filenames:
                file_path = Path(dirpath) / filename
                if str(file_path).startswith(str(self.dir_path / "imagesTr")):
                    self.training_files.append(file_path)
                elif str(file_path).startswith(str(self.dir_path / "labelsTr")):
                    self.label_files.append(file_path)
        self.files = list(zip(self.training_files, self.label_files))
        print(f"Found {len(self.training_files)} training files and {len(self.label_files)} label files.")
        print(f"Combined into {len(self.files)} pairs.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load the NIfTI images and labels
        image_file, label_file = self.files[idx]

        # Load NIfTI images using nibabel
        nifti_image = nib.load(image_file)
        nifti_label = nib.load(label_file)
        print(f"Loading image file: {image_file}")
        print(f"Loading label file: {label_file}")

        image_data = nifti_image.get_fdata()
        label_data = nifti_label.get_fdata()

        if self.transform:
            image_tensor, label_tensor = self.transform(image_data, label_data)

        pairs = []
        for i in range(len(image_tensor)):
            image = image_tensor[i].squeeze().unsqueeze(0)  # Remove the channel dimension
            label = label_tensor[i].squeeze().unsqueeze(0)  # Remove the channel dimension
            pairs.append((image, label))
        return pairs

if __name__ == "__main__":
    # Define the transformation
    crop_size = (96, 96, 64)
    crop_no = 10
    transform = NRandomCrop(crop_size, crop_no)

    # Create the dataset and dataloaders
    dataset = NIfTIDataset(dir_path=Path(__file__).parent / "Task03_Liver", transform=transform)

    