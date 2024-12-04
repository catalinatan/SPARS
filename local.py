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
    # Get the original image data
    img_data = nifti_image.get_fdata()

    # Calculate the zoom factors for each axis (original_shape / target_shape)
    zoom_factors = np.array(target_shape) / np.array(img_data.shape)

    # Resize the image data using scipy.ndimage.zoom
    resized_data = zoom(img_data, zoom_factors, order=1)  # Use order=1 for bilinear interpolation

    # Create a new NIfTI image with resized data and original affine
    resized_image = nib.Nifti1Image(resized_data, nifti_image.affine)

    return resized_image


class NIfTIDataset(Dataset):
    def __init__(self, dir_path):
        """
        Custom Dataset for loading NIfTI images from a tar file on a remote SSH server.
        Args:
            dir_path (str): Path to the folder on the remote server.
        """
        # Extract and process the tar file
        self.dir_path = dir_path
        self.files = []
        self._list_files_in_dir()

    def _list_files_in_dir(self):
        self.training_files = []
        self.label_files = []

        for dirpath, dirnames, filenames in os.walk(self.dir_path):
            for filename in filenames:
                file_path = Path(dirpath) / filename
                if str(file_path).startswith(os.path.join(self.dir_path, "imagesTr")):
                    self.training_files.append(file_path)
                elif str(file_path).startswith(os.path.join(self.dir_path , "labelsTr")):
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

        # Resize the images and labels
        resized_image = resize_image(nifti_image)
        resized_label = resize_image(nifti_label)

        # Convert resized images to numpy arrays and then to PyTorch tensors
        image_data = resized_image.get_fdata()
        label_data = resized_label.get_fdata()
        print(f"image shape: {image_data.shape}")
        print(f"label shape: {label_data.shape}")
        image_data = np.expand_dims(image_data, axis=0)
        label_data = np.expand_dims(label_data, axis=0)
        print(f"image shape: {image_data.shape}")
        print(f"label shape: {label_data.shape}")
        # Return as tensors
        return torch.tensor(image_data, dtype=torch.float32), torch.tensor(
            label_data, dtype=torch.float32
        )


# Define the split_dataset function
def split_dataset(
    dataset, train_ratio=0.5, val_ratio=0.2
):  # batch size of multiple of 8  (increase it until u are out of memory usually 64 or 96)
    # trade off between batch size and image size
    """
    Splits a dataset into training, validation, and test sets, and returns DataLoaders for each.

    Args:
        dataset (Dataset): The dataset to split.
        train_ratio (float): Proportion of data to use for training.
        val_ratio (float): Proportion of data to use for validation.
        batch_size (int): Batch size for the DataLoaders.

    Returns:
        train_loader, val_loader, test_loader: DataLoaders for training, validation, and testing.
    """
    # Compute sizes for each split
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    holdout_size = len(dataset) - train_size - val_size

    # Split the dataset
    train_data, val_data, holdout_data = random_split(
        dataset, [train_size, val_size, holdout_size]
    )

    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Holdout set size: {len(holdout_data)}")
    # Create DataLoaders
    train_loader = DataLoader(
        train_data, batch_size=32, shuffle=True
    )
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    holdout_loader = DataLoader(holdout_data, batch_size=32, shuffle=False)

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Holdout batches: {len(holdout_loader)}")
 
    return train_loader, val_loader, holdout_loader

# https://www.geeksforgeeks.org/building-a-convolutional-neural-network-using-pytorch/#step-3-define-the-cnn-architecture

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, stride=1, bias=True)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=2, bias=True)
        self.conv3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=2, bias=True)
        self.fc1 = nn.Linear(32 * 7 * 7 * 5, 64) # 245 is the number of features after flattening
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # for binary classification (1 for cancer, 0 for no cancer)
        # if accuracy is not enough, add convolution layers and fully connected layers
        # out_channels = max 96 but can increase to 64 

    def forward(self, x):
        # Print input shape
        print(f"Input shape: {x.shape}")
        # Apply conv1, relu activation, and pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply conv2, relu activation, and pooling
        print(f"Shape before conv2: {x.shape}")
        x = self.pool(F.relu(self.conv2(x)))
        # Apply conv3, relu activation, and pooling
        print(f"Shape before conv3: {x.shape}")
        x = self.pool(F.relu(self.conv3(x)))
        # Flatten for fully connected layers
        print(f"Shape before flattening: {x.shape}")
        x = x.view(x.size(0), -1) 
        # Apply fully connected layers with relu activation
        print(f"Shape after flattening: {x.shape}") # [32, 245] 32 is the batch size, 245 is the number of features
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Final layer (no activation for raw logits)
        x = self.fc3(x)
        return x


def train_network(net, train_loader, val_loader, criterion, optimizer):
    # Train the network
    for epoch in range(2):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            print(f"Batch {i}:")
            print(f"Inputs shape: {inputs.shape}")  # e.g., torch.Size([5, C, H, W]) or torch.Size([5, 256, 256, 180])
            print(f"Labels shape: {labels.shape}")  # e.g., torch.Size([5, ...])
            
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)  
            print(f"Output shape: {outputs.shape}")
            
            labels_list = []
            for sample_no in range(labels.shape[0]):
                single_label = labels[sample_no].long()
                if torch.max(single_label.flatten()) == 2:
                    labels_list.append(1)
                elif torch.max(single_label.flatten()) == 1:
                    labels_list.append(0)
            labels = torch.tensor(labels_list)
            print(labels)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(f"Epoch {epoch + 1}, Batch {i + 1} - Loss: {loss.item():.4f}")

        print(f"Testing after batch {i + 1}...")
        test_network(net, val_loader)
        
        torch.save(net.state_dict(), "model_weights.pth")
        print(f"Epoch {epoch + 1} - Average Training Loss: {running_loss / len(train_loader):.4f}")


def test_network(net, val_loader):
    net.eval()  # Set model to evaluation mode

    # Initialize confusion matrix counters
    true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            print(f"Output shape: {outputs.shape}")

            labels_list = []
            for sample_no in range(images.shape[0]):
                single_label = labels[sample_no].long()
                if torch.max(single_label.flatten()) == 2:
                    labels_list.append(1)
                elif torch.max(single_label.flatten()) == 1:
                    labels_list.append(0)
            labels = torch.tensor(labels_list)

            # Vectorized comparison
            matches = (predicted == labels)

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
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        sensitivity = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        specificity = true_negatives / (true_negatives + false_positives) if true_negatives + false_positives > 0 else 0
        
        # Print results
        print(f'Accuracy of the network: {accuracy:.2f} %')
        print('Confusion Matrix')
        print(f"Precision: {precision:.4f}")
        print(f"Sensitivity: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")

if __name__ == "__main__":
    dataset = NIfTIDataset(dir_path= "/raid/candi/catalina/Task03_Liver")

    # Split the dataset into training, validation and holdout sets
    train_loader, val_loader, holdout_loader = split_dataset(dataset)

    classes = ("cancer", "no_cancer")

    net = Net()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    # Train the network
    train_network(net, train_loader, val_loader, criterion, optimizer)

    # around 60-70% accuracy with good confusion matrix (as long as its not all in one class) then train to server 

    # in server, cd to raid/candi/catalina
    # follow install instructions https://github.com/conda-forge/miniforge
    # specify raid path to make .venv
    # activate source path/bin/activate
    # conda environment requirements txt file 
    # copy paste the code to the serve just change any path dependencies
    # run code in terminal python file name.py 