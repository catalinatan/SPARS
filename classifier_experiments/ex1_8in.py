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
from tqdm import tqdm
import skimage 

def resize_image(img_data):
    # Define target shape for the resized image
    target_shape = (256, 256, 180)

    # Convert input to a PyTorch tensor and add batch and channel dimensions
    img_tensor = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Resize using torch.nn.functional.interpolate (trilinear interpolation)
    resized_tensor = F.interpolate(img_tensor, size=target_shape, mode='trilinear', align_corners=False)

    resized_data = resized_tensor.squeeze() # Tensor Shape: (256, 256, 180)

    logging.info(f"Resized data shape: {resized_data.shape}")
    return resized_data

class NRandomCrop:
    def __init__(self, crop_size=(64, 64, 32), crop_no=10):  # Reduce number of crops
        self.crop_size = crop_size
        self.crop_no = crop_no

    def __call__(self, images, labels):
        labels[labels == 1] = 0
        labels[labels == 2] = 1

        labels = skimage.morphology.binary_dilation(labels)

        crops = []
        label_crops = []
        d, h, w = images.shape

        # Assuming labels is a 3D tensor with shape (512, 512, 72)
        labels = torch.tensor(labels)
        pos_crop_idx = torch.nonzero(labels == 1, as_tuple=False)
        neg_crop_idx = torch.nonzero(labels == 0, as_tuple=False) 
        # torch object with shape (n, 3) where n is the number of positive crops and 3 is the number of dimensions
        
        # Filter negative crop indexes to ensure that the crop size fits within the image
        neg_crop_idx[0] = np.clip(neg_crop_idx[0], 0, d - self.crop_size[0] - 1)
        neg_crop_idx[1] = np.clip(neg_crop_idx[1], 0, h - self.crop_size[1] - 1)
        neg_crop_idx[2] = np.clip(neg_crop_idx[2], 0, w - self.crop_size[2] - 1)
       
        if len(pos_crop_idx) == 0 and len(neg_crop_idx) >= self.crop_no:
            for i in range(self.crop_no):
                random_neg_idx = random.choice(neg_crop_idx)

                d_start, h_start, w_start = random_neg_idx

                crop = images[d_start:d_start + self.crop_size[0],
                            h_start:h_start + self.crop_size[1],
                            w_start:w_start + self.crop_size[2]]
                label_crop = labels[d_start:d_start + self.crop_size[0],
                                    h_start:h_start + self.crop_size[1],
                                    w_start:w_start + self.crop_size[2]]

                resized_crop = resize_image(crop)
                resized_label_crop = resize_image(label_crop)

                crops.append(resized_crop)
                label_crops.append(resized_label_crop)
            
        elif len(pos_crop_idx) and len(neg_crop_idx):
            for i in range(int(self.crop_no / 2)):
                random_pos_idx = random.choice(pos_crop_idx)

                d_start, h_start, w_start = random_pos_idx

                crop = images[d_start:d_start + self.crop_size[0],
                            h_start:h_start + self.crop_size[1],
                            w_start:w_start + self.crop_size[2]]
                
                label_crop = labels[d_start:d_start + self.crop_size[0],
                                    h_start:h_start + self.crop_size[1],
                                    w_start:w_start + self.crop_size[2]]

                resized_crop = resize_image(crop)
                resized_label_crop = resize_image(label_crop)

                crops.append(resized_crop)
                label_crops.append(resized_label_crop)

            for i in range(int(self.crop_no / 2)):
                random_neg_idx = random.choice(neg_crop_idx)

                d_start, h_start, w_start = random_neg_idx

                crop = images[d_start:d_start + self.crop_size[0],
                            h_start:h_start + self.crop_size[1],
                            w_start:w_start + self.crop_size[2]]
                
                label_crop = labels[d_start:d_start + self.crop_size[0],
                                    h_start:h_start + self.crop_size[1],
                                    w_start:w_start + self.crop_size[2]]

                resized_crop = resize_image(crop)
                resized_label_crop = resize_image(label_crop)

                crops.append(resized_crop)
                label_crops.append(resized_label_crop)

        resized_crops = [crop.unsqueeze(0).unsqueeze(0) for crop in crops]
        new_resized_crops = np.concatenate(resized_crops, axis=0)

        resized_label_crops = [label_crop.unsqueeze(0).unsqueeze(0) for label_crop in label_crops]
        new_resized_label_crops = np.concatenate(resized_label_crops, axis=0)
        
        print(f"Resized crops shape: {new_resized_crops.shape}")
        print(f"Resized label crops shape: {new_resized_label_crops.shape}")

        return new_resized_crops, new_resized_label_crops
    
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

class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label
    
class NIfTIDataset():
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
        patient_names = [element for element in patient_names if not element.startswith(".")]
        patient_names = [element for element in patient_names if not element.startswith("._")]

        self.label_files = [os.path.join(self.dir_path, "labelsTr", element) for element in patient_names]
        self.training_files = [element.replace("labelsTr", "imagesTr") for element in self.label_files]

        self.files = list(zip(self.training_files, self.label_files))
        print(f"Found {len(self.training_files)} training files and "
              f"{len(self.label_files)} label files.")
        print(f"Combined into {len(self.files)} pairs.")
    def __len__(self):
        return len(self.files)
    
    def get_data_batch(self, batch_size, start_file_no=0, end_file_no=1):

        images_list = []
        labels_list = []

        print(f"Length of files: {len(self.files)}")

        self.limited_files = self.files[start_file_no:end_file_no]
        print(f"Length of limited files: {len(self.limited_files)}")
    
        for i in range(batch_size):
            # Load the NIfTI images and labels
            idx = random.randint(0, len(self.limited_files) - 1)
            image_file, label_file = self.limited_files[idx]

            # Load NIfTI images using nibabel
            nifti_image = nib.load(image_file)
            nifti_label = nib.load(label_file)

            image_data = nifti_image.get_fdata()
            label_data = nifti_label.get_fdata()

            if self.transform:
                image_tensors, labels = self.transform(image_data, label_data)

            logging.info(f"Image tensor shape{image_tensors.shape}")
            logging.info(f"Label shape {labels.shape}")

            images_list.append(image_tensors)
            labels_list.append(labels)

            print(f"iter: {i}")

        concatenated_imgs = np.concatenate(images_list, axis=0)
        concatenated_labels = np.concatenate(labels_list, axis=0)

        print(f"Shape of concatenated images: {concatenated_imgs.shape}")
        print(f"Shape for label list: {concatenated_labels.shape}")

        return torch.tensor(concatenated_imgs), torch.tensor(concatenated_labels)
    
    
def train_network(net, dataset_object, criterion, optimizer):
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

    no_of_batches = 2
    start_file_no = 0
    end_file_no = 8
    batch_size = 4

    for epoch in range(32): 
        net.train()
        running_loss = 0.0

        print('epoch:', epoch + 1)

        for i in tqdm(range(no_of_batches), desc="Training Progress"):
            inputs, labels = dataset_object.get_data_batch(batch_size, start_file_no, end_file_no)

            labels = torch.amax(labels, dim=(2, 3, 4))
            
            # Flatten the labels to 1D tensor
            labels = labels.view(-1)
            
            # Ensure labels are class indices (0 or 1)
            labels = labels.long()

            print(f"inputs shape: {inputs.shape}")
            print(f"labels shape: {labels.shape}")
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)

            print(f"outputs shape: {outputs.shape}")
            # Compute the loss
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            (
                f"Epoch {epoch + 1} - Average Training Loss: "
                f"{running_loss / no_of_batches:.4f}"
            )
        )

        print("Evaluating the network")
        # Evaluate the network on the validation data
        test_network(net, dataset_object)
        # Save the model weights
        torch.save(net.state_dict(), "ex1_8in_weights.pth")


def test_network(net, dataset_object):
    net.eval()  # Set model to evaluation mode

    # Initialize confusion matrix counters
    true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0

    no_of_batches = 4
    start_file_no = 8
    end_file_no = 24
    batch_size = 4

    with torch.no_grad():
        for i in range(no_of_batches):
            images, labels = dataset_object.get_data_batch(batch_size, start_file_no, end_file_no)

            labels = torch.amax(labels, dim=(2, 3, 4))
            
            # Flatten the labels to 1D tensor
            labels = labels.view(-1)
            
            # Convert to long type 
            labels = labels.long()

            outputs = net(images)

            __, predicted = torch.max(outputs.data, 1)

            matches = (predicted == labels)
            correct = matches.sum().item()
            total = len(labels)

            true_positives += ((predicted == 1) & (labels == 1)).sum().item()
            true_negatives += ((predicted == 0) & (labels == 0)).sum().item()
            false_positives += ((predicted == 1) & (labels == 0)).sum().item()
            false_negatives += ((predicted == 0) & (labels == 1)).sum().item()

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

        print(f"true_positives: {true_positives}")
        print(f"true_negatives: {true_negatives}")
        print(f"false_positives: {false_positives}")
        print(f"false_negatives: {false_negatives}")
        print(f"Accuracy: {correct / total:.2f}")
        print("Confusion Matrix")
        print(f"Precision: {precision:.4f}")
        print(f"Sensitivity: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")

if __name__ == "__main__":
    # Define the transformation
    crop_size = (96, 96, 64)
    crop_no = 10
    transform = NRandomCrop(crop_size, crop_no)

    dir_path="/raid/candi/catalina/Task03_Liver"

    # Initialize the dataset
    dataset = NIfTIDataset(dir_path, transform=transform)

    print("Data set loaded")
    # Define the class labels
    classes = ("no cancer", "cancer")

    # Initialize the neural network
    net = Net()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # Use Adam optimizer

    # Train the network
    train_network(net, dataset, criterion, optimizer)