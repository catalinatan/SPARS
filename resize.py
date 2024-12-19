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
    def __init__(self, crop_size=(96, 96, 64), crop_no=200):
        self.crop_size = crop_size
        self.crop_no = crop_no

    def __call__(self, images, labels):
        crops = []
        label_crops = []
        d, h, w = images.shape  # Assuming images shape is (D, H, W)

        for _ in range(self.crop_no):
            d_start = random.randint(0, d - self.crop_size[0])
            h_start = random.randint(0, h - self.crop_size[1])
            w_start = random.randint(0, w - self.crop_size[2])

            crop = images[d_start:d_start + self.crop_size[0],
                          h_start:h_start + self.crop_size[1],
                          w_start:w_start + self.crop_size[2]]
            label_crop = labels[d_start:d_start + self.crop_size[0],
                                h_start:h_start + self.crop_size[1],
                                w_start:w_start + self.crop_size[2]]
            crops.append(crop)
            label_crops.append(label_crop)

        print(f"no of cropped images: {len(crops)}")
        print(f"no of cropped labels: {len(label_crops)}")
        print(f"cropped image shape: {crops[0].shape}")
        print(f"cropped label shape: {label_crops[0].shape}")

        # Resize each crop and label crop
        resized_crops = []
        resized_label_crops = []
        for crop, label_crop in zip(crops, label_crops):
            resized_crops.append(resize_image(crop))
            resized_label_crops.append(resize_image(label_crop))

        print(f"resized image shape: {resized_crops[0].shape}")
        print(f"resized label shape: {resized_label_crops[0].shape}")
        return torch.stack([crop.clone().detach() for crop in resized_crops]), \
               torch.stack([label_crop.clone().detach() for label_crop in resized_label_crops])


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


# Define the split_dataset function
def split_dataset(
    dataset, train_ratio=0.5, val_ratio=0.2, batch_size=8, num_workers=0, pin_memory=True
):
    # batch size of multiple of 8
    # (increase it until you are out of memory usually 64 or 96)
    # trade off between batch size and image size
    """
    Splits a dataset into training, validation, and test sets, and returns
    DataLoaders for each.

    Args:
        dataset (Dataset): The dataset to split.
        train_ratio (float): Proportion of data to use for training.
        val_ratio (float): Proportion of data to use for validation.
        batch_size (int): Batch size for the DataLoaders.

    Returns:
        train_loader, val_loader, test_loader: DataLoaders for training,
        validation, and testing.
    """
    # Compute sizes for each split
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    holdout_size = len(dataset) - train_size - val_size

    # Split the dataset
    train_data, val_data, holdout_data = random_split(
        dataset, [train_size, val_size, holdout_size]
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    holdout_loader = DataLoader(
        holdout_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")
    print(f"Holdout set size: {holdout_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Holdout batches: {len(holdout_loader)}")

    return train_loader, val_loader, holdout_loader


# https://www.geeksforgeeks.org/building-a-convolutional-neural-network-using-pytorch/#step-3-define-the-cnn-architecture

class Net(nn.Module):
    """
    A 3D Convolutional Neural Network for binary classification.

    This network consists of three convolutional layers followed by
    max pooling layers, and 5 fully connected layers.
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels=1, out_channels=8, kernel_size=3, stride=1, bias=True
        )
        self.bn1 = nn.BatchNorm3d(8)
        self.conv2 = nn.Conv3d(
            in_channels=8, out_channels=16, kernel_size=3, stride=2, bias=True
        )
        self.bn2 = nn.BatchNorm3d(16)
        self.conv3 = nn.Conv3d(
            in_channels=16, out_channels=32, kernel_size=3, stride=2, bias=True
        )
        self.bn3 = nn.BatchNorm3d(32)
        self.conv4 = nn.Conv3d(
            in_channels=32, out_channels=64, kernel_size=3, stride=2, bias=True
        )
        self.bn4 = nn.BatchNorm3d(64)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 3 * 3 * 2, 1600)
        self.fc2 = nn.Linear(1600, 800)
        self.fc3 = nn.Linear(800, 400)
        self.fc4 = nn.Linear(400, 128)
        self.fc5 = nn.Linear(128, 2)
        # if accuracy is not enough, add convolution layers
        # and fully connected layers
        # out_channels = max 96 but can increase to 64

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, 2).
        """
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) 
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    

def train_network(net, train_loader, val_loader, criterion, optimizer):
    for epoch in range(2):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        
        for batch in train_loader:
            print(f"Batch shape: {len(batch)}")
            print(f"file shape: {len(batch[0])}")
            for file in batch:
                for inputs, labels in file:
                    print(f"inputs: {inputs.shape}"
                          f"labels: {labels.shape}")
            # `data` shape: [100, 5, 1, 256, 256, 180]
            # Rearrange and merge the first two dimensions
            inputs = inputs.permute(1, 0, 2, 3, 4, 5)  # Move "file" dimension to the front, shape: [5, 100, 1, 256, 256, 180]
            data = data.reshape(-1, *data.shape[2:])  # Merge first two dimensions, shape: [500, 1, 256, 256, 180]

        #     optimizer.zero_grad()

        #     outputs = net(inputs)
        #     print(f"Output shape: {outputs.shape}")

        #     max_labels = torch.max(labels.flatten(start_dim=1), dim=1)[0]
        #     labels = torch.where(max_labels == 2, torch.tensor(1), torch.tensor(0))
        #     print(f"Converted labels: {labels}")

        #     loss = criterion(outputs, labels)
        #     loss.backward()
        #     optimizer.step()

        #     running_loss += loss.item()
        #     print(f"Epoch {epoch + 1}, Batch {i + 1} - Loss: {loss.item():.4f}")

        # print(f"Epoch {epoch + 1} - Average Training Loss: {running_loss / len(train_loader):.4f}")

if __name__ == "__main__":
    # Define the transformation
    crop_size = (96, 96, 64)
    crop_no = 200
    transform = NRandomCrop(crop_size, crop_no)

    # Create the dataset and dataloaders
    dataset = NIfTIDataset(dir_path=Path(__file__).parent / "Task03_Liver", transform=transform)

    # Load the first sample
    pairs = dataset[0]
    print(f"Number of pairs: {len(pairs)}")
    print(f"First pair - Image shape: {pairs[0][0].shape}, Label shape: {pairs[0][1].shape}")

    # Split the dataset into training, validation and holdout sets
    train_loader, val_loader, holdout_loader = split_dataset(dataset)

    classes = ("cancer", "no_cancer")

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    # Train the network
    train_network(net, train_loader, val_loader, criterion, optimizer)