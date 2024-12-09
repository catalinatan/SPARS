import nibabel as nib
import os
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torch
from pathlib import Path
from scipy.ndimage import zoom
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def resize_image(nifti_image, target_shape=(256, 256, 180)):
    """
    Resizes a NIfTI image to the target shape using bilinear interpolation.

    Args:
        nifti_image (nib.Nifti1Image): The NIfTI image to resize.
        target_shape (tuple): The target shape for the resized image.

    Returns:
        nib.Nifti1Image: The resized NIfTI image with the original affine.
    """
    # Get the original image data as a numpy array
    img_data = nifti_image.get_fdata()

    # Calculate the zoom factors for each axis (target_shape / original_shape)
    zoom_factors = np.array(target_shape) / np.array(img_data.shape)

    # Resize the image data using scipy.ndimage.zoom
    # with bilinear interpolation (order=1)
    resized_data = zoom(img_data, zoom_factors, order=1)

    # Create a new NIfTI image with the resized data and
    # the original affine transformation
    resized_image = nib.Nifti1Image(resized_data, nifti_image.affine)

    return resized_image


class NIfTIDataset(Dataset):
    def __init__(self, dir_path):
        """
        Custom Dataset for loading NIfTI images from a directory.

        Args:
            dir_path (str): Path to the folder containing NIfTI images.
        """
        self.dir_path = dir_path
        self.files = []
        self._list_files_in_dir()

    def _list_files_in_dir(self):
        """
        Lists and pairs the training and label files in the directory.

        This method populates the training_files and label_files lists
        and pairs them into the files attribute.
        """
        self.training_files = []
        self.label_files = []

        for dirpath, _, filenames in os.walk(self.dir_path):
            for filename in filenames:
                if filename.startswith("._"):
                    continue
                file_path = Path(dirpath) / filename
                if "imagesTr" in file_path.parts:
                    self.training_files.append(file_path)
                elif "labelsTr" in file_path.parts:
                    self.label_files.append(file_path)

        self.files = list(zip(self.training_files, self.label_files))

        print(f"Found {len(self.training_files)} training files and "
              f"{len(self.label_files)} label files.")
        print(f"Combined into {len(self.files)} pairs.")

    def __len__(self):
        """
        Returns the number of file pairs in the dataset.

        Returns:
            int: Number of file pairs.
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        Loads and returns a pair of NIfTI images and labels as tensors.

        Args:
            idx (int): Index of the file pair to load.

        Returns:
            tuple: A tuple containing the image tensor and the label tensor.
        """
        # Load the NIfTI images and labels
        image_file, label_file = self.files[idx]

        # Load NIfTI images using nibabel
        nifti_image = nib.load(image_file)
        nifti_label = nib.load(label_file)

        # Resize the images and labels
        resized_image = resize_image(nifti_image)
        resized_label = resize_image(nifti_label)

        # Convert resized images to numpy arrays and then to PyTorch tensors
        image_data = np.expand_dims(resized_image.get_fdata(), axis=0)
        label_data = np.expand_dims(resized_label.get_fdata(), axis=0)

        # Return as tensors
        image_tensor = torch.tensor(image_data, dtype=torch.float32)
        label_tensor = torch.tensor(label_data, dtype=torch.float32)
        return image_tensor, label_tensor


def split_dataset(
    dataset, train_ratio=0.5, val_ratio=0.2, batch_size=56
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
    for epoch in range(2):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            print(f"Batch {i}:")
            print(f"Inputs shape: {inputs.shape}")
            print(f"Labels shape: {labels.shape}")

            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)

            # Along the last dimension
            max_labels = torch.max(labels.flatten(start_dim=1), dim=1)[0]

            # Create a new tensor with 1 where max_labels is 2,
            # and 0 where max_labels is 1
            labels = torch.where(
                max_labels == 2, torch.tensor(1), torch.tensor(0)
            )

            print(f"Training labels: {labels}")
            # Compute the loss
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
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
            print(f"Output shape: {outputs.shape}")

            # Along the last dimension
            max_labels = torch.max(labels.flatten(start_dim=1), dim=1)[0]

            # Create a new tensor with 1 where max_labels is 2,
            # and 0 where max_labels is 1
            labels = torch.where(
                max_labels == 2, torch.tensor(1), torch.tensor(0)
            )

            print(f"Predicted labels: {predicted}")
            print(f"Validation labels: {labels}")
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
        print(f"Accuracy of the network: {accuracy:.2f} %")
        print("Confusion Matrix")
        print(f"True Positives: {true_positives}")
        print(f"True Negatives: {true_negatives}")
        print(f"False Positives: {false_positives}")
        print(f"False Negatives: {false_negatives}")
        print(f"Precision: {precision:.4f}")
        # Print sensitivity
        if sensitivity != 0:
            print(f"Sensitivity: {sensitivity:.4f}")
        else:
            print("No true positives")
        # Print specificity
        if specificity != 0:
            print(f"Specificity: {specificity:.4f}")
        else:
            print("No true negatives")

# https://saturncloud.io/blog/how-to-use-class-weights-with-focal-loss-in-pytorch-for-imbalanced-multiclass-classification/#use-class-weights


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss

# https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
def calculate_weights(train_loader):
    pos_labels = 0
    neg_labels = 0
    tot_labels = 0

    for _, labels in train_loader:
        max_labels = torch.max(labels.flatten(start_dim=1), dim=1)[0]
        pos_labels += (max_labels == 2).sum().item()
        neg_labels += (max_labels == 1).sum().item()
        tot_labels += len(max_labels)

    print(f"Positive labels (1): {pos_labels}")
    print(f"Negative labels (0): {neg_labels}")
    print(f"Total labels: {tot_labels}")

    # Calculate unnormalized weights
    pos_weight = (tot_labels / (pos_labels * 2)) if pos_labels > 0 else 1
    neg_weight = (tot_labels / (neg_labels * 2)) if neg_labels > 0 else 1

    # Normalize weights
    total_weight = pos_weight + neg_weight
    pos_weight_normalized = pos_weight / total_weight
    neg_weight_normalized = neg_weight / total_weight
    
    class_weights = torch.tensor([neg_weight_normalized, pos_weight_normalized], dtype=torch.float32)
    print(f"Normalized class weights: {class_weights}")
    return class_weights


if __name__ == "__main__":
    # Initialize the dataset
    dataset = NIfTIDataset(dir_path="/raid/candi/catalina/Task03_Liver")

    # Split the dataset into training, validation, and holdout sets
    train_loader, val_loader, holdout_loader = split_dataset(dataset)

    # Define the class labels
    classes = ("no cancer", "cancer")

    # Initialize the neural network
    net = Net()

    # Define class weights
    # class_weights = calculate_weights(train_loader)
    class_weights = torch.tensor([0.8, 0.2], dtype=torch.float32)
    # Define the loss function and optimizer
    criterion = FocalLoss(alpha=class_weights, gamma=2)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train the network
    train_network(net, train_loader, val_loader, criterion, optimizer)
