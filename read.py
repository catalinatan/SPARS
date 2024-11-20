import paramiko
import tarfile
import nibabel as nib
import os
import tempfile
from nilearn.image import resample_img
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torch

"""References:
Paramiko and SSH:
https://stackoverflow.com/questions/50457085/file-transfer-with-tar-piped-through-ssh-using-python
https://medium.com/@keagileageek/paramiko-how-to-ssh-and-file-transfers-with-python-75766179de73
"""


def resize_image(nifti_image):
    target_affine = np.eye(3)  # Identity matrix for affine transformation
    target_shape = (256, 256, 90)
    resampled_image = resample_img(nifti_image, target_affine=target_affine, target_shape=target_shape, interpolation='nearest')
    return resampled_image


class NIfTIDataset(Dataset):
    def __init__(self, ssh_host, ssh_username, ssh_password, tar_path, target_directory, transform=None):
        """
        Custom Dataset for loading NIfTI images from a tar file on a remote SSH server.
        Args:
            ssh_host (str): SSH server hostname or IP address.
            ssh_username (str): SSH username.
            ssh_password (str): SSH password.
            tar_path (str): Path to the tar file on the remote server.
            target_directory (str): Directory within the tar file containing the NIfTI images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Initialize SSH and SFTP clients
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh_client.connect(hostname=ssh_host, username=ssh_username, password=ssh_password)
        self.sftp_client = self.ssh_client.open_sftp()

        # Extract and process the tar file
        self.files = []
        self.tar_path = tar_path
        self.target_directory = target_directory
        self._list_files_in_tar()

    def _list_files_in_tar(self):
        # Read the tar file remotely using SFTP
        with self.sftp_client.open(self.tar_path, 'rb') as remote_tar_file:
            with tarfile.open(fileobj=remote_tar_file, mode='r') as tar:
                for member in tar.getmembers():
                    if member.name.startswith(self.target_directory) and not os.path.basename(member.name).startswith('._'):
                        self.files.append(member)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        member = self.files[idx]

        # Read the file directly from the tar archive
        with self.sftp_client.open(self.tar_path, 'rb') as remote_tar_file:
            with tarfile.open(fileobj=remote_tar_file, mode='r') as tar:
                extracted_file = tar.extractfile(member)
                # Write the extracted file to a temporary file
                with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as temp_file:
                    temp_file.write(extracted_file.read())
                    temp_file_path = temp_file.name

                # Load the NIfTI image using nibabel
                nifti_image = nib.load(temp_file_path)

                # Resize the image
                resized_image = resize_image(nifti_image)

                # Convert the resized image to numpy array before returning
                image_data = resized_image.get_fdata()

                if self.transform:
                    image_data = self.transform(image_data)

                # Clean up the temporary file
                os.remove(temp_file_path)

                return torch.tensor(image_data, dtype=torch.float32), os.path.basename(member.name)

    def __del__(self):
        # Close the SFTP and SSH clients
        self.sftp_client.close()
        self.ssh_client.close()


def split_dataset(sftp, target_directory):
    # Define split sizes (e.g., 50% for training, 20% for validation, 30% for holdout)
    train_size = int(0.5 * len(dataset))
    val_size = int(0.2 * len(dataset))
    holdout_size = len(dataset) - train_size - val_size

    # Split the dataset into 3 parts
    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, holdout_size])

    # Create DataLoaders
    batch_size = 4  # Adjust as needed
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    folder_names = ['training_images', 'validation_images', 'holdout_images']
    for folder in folder_names:
        folder_path = os.path.join(target_directory, 'resized_images', folder)
        sftp.mkdir(folder_path)
    torch.utils.data.random_split()


if __name__ == "__main__":
    dataset = NIfTIDataset(
        ssh_host='128.16.4.13',
        ssh_username='catalina',
        ssh_password='secret',
        tar_path='/raid/candi/catalina/Task03_Liver.tar',
        target_directory='Task03_Liver/imagesTr'
    )

