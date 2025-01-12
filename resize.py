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
import logging

# Configure logging
logging.basicConfig(filename='debug.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def resize_image(img_data):
    # Define target shape for the resized image
    target_shape = (256, 256, 180)

    # Convert input to a PyTorch tensor and add batch and channel dimensions
    img_tensor = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, D, H, W)

    # Resize using torch.nn.functional.interpolate (trilinear interpolation)
    resized_tensor = F.interpolate(img_tensor, size=target_shape, mode='trilinear', align_corners=False)

    resized_data = resized_tensor.squeeze() # Tensor Shape: (256, 256, 180)

    logging.info(f"Resized data shape: {resized_data.shape}")
    return resized_data


class NRandomCrop:
    def __init__(self, crop_size=(64, 64, 32), crop_no=8):  # Reduce number of crops
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

            # Adjust starting indices to ensure valid crop size
            if d_start + self.crop_size[0] > d:
                d_start = d - self.crop_size[0]
            if h_start + self.crop_size[1] > h:
                h_start = h - self.crop_size[1]
            if w_start + self.crop_size[2] > w:
                w_start = w - self.crop_size[2]

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

        logging.info(f"Resized crops: {len(resized_crops)}")
        logging.info(f"Relabeled labels: {len(relabeled_labels)}")

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

        logging.info(f"Directory path: {self.dir_path}")  # Debug statement

        if not os.path.exists(self.dir_path):
            logging.error(f"Directory does not exist: {self.dir_path}")  # Debug statement
            return

        patient_names = os.listdir(os.path.join(self.dir_path, "labelsTr"))
        patient_names = [element for element in patient_names if not element.startswith("._")]
        patient_names = patient_names[:10]

        self.label_files = [os.path.join(self.dir_path, "labelsTr", element) for element in patient_names]
        self.training_files = [os.path.join(self.dir_path, "imagesTr", element) for element in patient_names]

        self.files = list(zip(self.training_files, self.label_files))

        logging.info(f"Found {len(self.training_files)} training files and "
              f"{len(self.label_files)} label files.")
        logging.info(f"Combined into {len(self.files)} pairs.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        logging.info(f"Fetching item index: {idx}")  # Debug statement

        # Load the NIfTI images and labels
        image_file, label_file = self.files[idx]

        # Load NIfTI images using nibabel
        nifti_image = nib.load(image_file)
        nifti_label = nib.load(label_file)

        image_data = nifti_image.get_fdata()
        label_data = nifti_label.get_fdata()

        if self.transform:
            image_tensor, label = self.transform(image_data, label_data)

        pairs = []
        for i in range(len(image_tensor)):
            image = image_tensor[i].squeeze().unsqueeze(0)  # Remove the channel dimension

            pairs.append((image, label))
        return pairs

def split_dataset(
    dataset, train_ratio=0.5, val_ratio=0.2, batch_size=32
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
        train_data, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False
    )
    holdout_loader = DataLoader(
        holdout_data, batch_size=batch_size, shuffle=False
    )

    logging.info(f"Training set size: {train_size}")
    logging.info(f"Validation set size: {val_size}")
    logging.info(f"Holdout set size: {holdout_size}")
    logging.info(f"Training batches: {len(train_loader)}")
    logging.info(f"Validation batches: {len(val_loader)}")
    logging.info(f"Holdout batches: {len(holdout_loader)}")

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
    """
    Trains the network on the training data and evaluates on the
    validation data.

    Args:
        net (nn.Module): The neural network to train.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating the
        network weights.

    Returns:
        None
    """
    for epoch in range(2):  # Reduce number of epochs
        net.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            logging.info(f"Batch {i}: Inputs shape: {inputs.shape}, Training labels: {labels}")

            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        logging.info(
            (
                f"Epoch {epoch + 1} - Average Training Loss: "
                f"{running_loss / len(train_loader):.4f}"
            )
        )

        # Evaluate the network on the validation data
        test_network(net, val_loader)

        # Save the model weights
        torch.save(net.state_dict(), "model_weights.pth")


def test_network(net, val_loader):
    """
    Evaluates the network on the validation data and calculates
    performance metrics.

    Args:
        net (nn.Module): The neural network to evaluate.
        val_loader (DataLoader): DataLoader for the validation data.

    Returns:
        None
    """
    net.eval()  # Set model to evaluation mode

    # Initialize confusion matrix counters
    true_positives, true_negatives = 0, 0
    false_positives, false_negatives = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            logging.info(f"Output shape: {outputs.shape}, Predicted labels: {predicted}, Validation labels: {labels}")
            
            # Vectorized comparison
            matches = predicted == labels

            # Count correct predictions
            correct = matches.sum().item()
            total = len(labels)

            # Confusion matrix calculations (vectorized)
            true_positives += ((predicted == 1) & (labels == 1)).sum().item()
            true_negatives += ((predicted == 0) & (labels == 0)).sum().item()
            false_positives += ((predicted == 1) & (labels == 0)).sum().item()
            false_negatives += ((predicted == 0) & (labels == 1)).sum().item()

        # Calculate metrics
        accuracy = 100 * correct / total
        precision = (
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives > 0
            else 0
        )
        sensitivity = (
            true_positives / (true_positives + false_negatives)
            if true_positives + false_negatives > 0
            else 0
        )
        specificity = (
            true_negatives / (true_negatives + false_positives)
            if true_negatives + false_positives > 0
            else 0
        )

        # Print results
        logging.info(f"Accuracy of the network: {accuracy:.2f} %")
        logging.info("Confusion Matrix")
        logging.info(f"True Positives: {true_positives}")
        logging.info(f"True Negatives: {true_negatives}")
        logging.info(f"False Positives: {false_positives}")
        logging.info(f"False Negatives: {false_negatives}")
        logging.info(f"Precision: {precision:.4f}")
        # Print sensitivity
        if sensitivity != 0:
            logging.info(f"Sensitivity: {sensitivity:.4f}")
        else:
            logging.info("No true positives")
        # Print specificity
        if specificity != 0:
            logging.info(f"Specificity: {specificity:.4f}")
        else:
            logging.info("No true negatives")

if __name__ == "__main__":
    # Define the transformation
    crop_size = (96, 96, 64)
    crop_no = 10
    transform = NRandomCrop(crop_size, crop_no)

    # Initialize the dataset
    dataset = NIfTIDataset(dir_path="/raid/candi/catalina/Task03_Liver", transform=transform)

    # Split the dataset into training, validation, and holdout sets
    train_loader, val_loader, holdout_loader = split_dataset(dataset)

    # Define the class labels
    classes = ("no cancer", "cancer")

    # Initialize the neural network
    net = Net()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # Use Adam optimizer

    # Train the network
    train_network(net, train_loader, val_loader, criterion, optimizer)