import os
import torch
import nibabel as nib
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random


class NIfTIDataset(Dataset):
    def __init__(self, dir_path, transform):
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
        print(f"Combined into {len(self.files)} pairs.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_file, label_file = self.files[idx]

        print(f"Loading image file: {image_file}")
        print(f"Loading label file: {label_file}")
        print(f"data type of image file: {type(image_file)}")
        print(f"data type of label file: {type(label_file)}")
        
        nifti_image = nib.load(str(image_file))
        nifti_label = nib.load(str(label_file))

        image_data = nifti_image.get_fdata()
        label_data = nifti_label.get_fdata()

        image_tensor, label_tensor = self.transform(image_data, label_data)

        pairs = []
        for i in range(len(image_tensor)):
            image = image_tensor[i].squeeze().unsqueeze(0)
            label = label_tensor[i].squeeze().unsqueeze(0)
            pairs.append((image, label))

        return pairs


def resize_image(img_data):
    target_shape = (256, 256, 180)
    img_tensor = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    resized_tensor = F.interpolate(img_tensor, size=target_shape, mode='trilinear', align_corners=False)
    resized_data = resized_tensor.squeeze()
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


def pre_save_crops(dataset, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for idx, (resized_crops, relabeled_labels) in enumerate(dataset):
        for crop_idx, crop in enumerate(resized_crops):
            # Save each crop as a .pt file
            crop_save_path = os.path.join(save_dir, f"image_{idx}_{crop_idx}.pt")
            label_save_path = os.path.join(save_dir, f"label_{idx}_{crop_idx}.pt")
            
            # Save tensor data and integer label
            torch.save(crop, crop_save_path)
            torch.save(relabeled_labels[crop_idx], label_save_path)


if __name__ == "__main__":
    crop_size = (96, 96, 64)
    crop_no = 10
    transform = NRandomCrop(crop_size, crop_no)

    dataset = NIfTIDataset(dir_path=Path(__file__).parent / "Task03_Liver", transform=transform)
    print(f"Dataset length: {len(dataset)}")