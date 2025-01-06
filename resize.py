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

    print(f"Resized data shape: {resized_data.shape}")
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

            print(f"Crop shape: {crop.shape}")
            print(f"Label crop shape: {label_crop.shape}")

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
        """
        Lists and pairs the training and label files in the directory.

        This method populates the training_files and label_files lists
        and pairs them into the files attribute.
        """
        self.training_files = []
        self.label_files = []

        print(f"Directory path: {self.dir_path}")  # Debug statement

        if not os.path.exists(self.dir_path):
            print(f"Directory does not exist: {self.dir_path}")  # Debug statement
            return

        for dirpath, _, filenames in os.walk(self.dir_path):
            print(f"Checking directory: {dirpath}")  # Debug statement
            for filename in filenames:
                print(f"Found file: {filename}")  # Debug statement
                if filename.startswith("._"):
                    continue
                file_path = Path(dirpath) / filename
                if "imagesTr" in file_path.parts:
                    print(f"Found image file: {file_path}")
                    self.training_files.append(file_path)
                elif "labelsTr" in file_path.parts:
                    print(f"Found label file: {file_path}")
                    self.label_files.append(file_path)

        self.files = list(zip(self.training_files, self.label_files))
        print(f"self.files: {self.files}")

        print(f"Found {len(self.training_files)} training files and "
              f"{len(self.label_files)} label files.")
        print(f"Combined into {len(self.files)} pairs.")


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        print(f"Fetching item index: {idx}")  # Debug statement

        # Load the NIfTI images and labels
        image_file, label_file = self.files[idx]

        # Load NIfTI images using nibabel
        nifti_image = nib.load(image_file)
        nifti_label = nib.load(label_file)
        print(f"Loading image file: {image_file}")
        print(f"Loading label file: {label_file}")

        image_data = nifti_image.get_fdata()
        label_data = nifti_label.get_fdata()

        print(f"Image shape: {image_data.shape}")
        print(f"Label shape: {label_data.shape}")
        
        if self.transform:
            print("Applying transform")  # Debug statement
            image_tensor, label_tensor = self.transform(image_data, label_data)
        else:
            image_tensor = torch.tensor(image_data, dtype=torch.float32).unsqueeze(0)
            label_tensor = torch.tensor(label_data, dtype=torch.float32).unsqueeze(0)

        pairs = []
        for i in range(len(image_tensor)):
            image = image_tensor[i].squeeze().unsqueeze(0)  # Remove the channel dimension
            label = label_tensor[i].squeeze().unsqueeze(0)  # Remove the channel dimension
            pairs.append((image, label))
            print(f"Image shape: {image.shape}")
            print(f"Label shape: {label.shape}")
        return pairs

if __name__ == "__main__":
    # Define the transformation
    crop_size = (96, 96, 64)
    crop_no = 10
    transform = NRandomCrop(crop_size, crop_no)

    # Initialize the dataset
    dataset = NIfTIDataset(dir_path="/raid/candi/catalina/Task03_Liver", transform=transform)
    
    #dir_path = Path(__file__).parent / "Task03_Liver"
    #print(f"{dir_path}")

    # Create the dataset and dataloaders
   # dataset = NIfTIDataset(dir_path=Path(__file__).parent / "Task03_Liver", transform=transform)