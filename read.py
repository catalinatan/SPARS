import paramiko
import tarfile
import nibabel as nib
import os
import tempfile
from nilearn.image import resample_img
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
import torch 

"""References:
Paramiko and SSH:
https://stackoverflow.com/questions/50457085/file-transfer-with-tar-piped-through-ssh-using-python
https://medium.com/@keagileageek/paramiko-how-to-ssh-and-file-transfers-with-python-75766179de73
"""


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
            # Open the tar file using the tarfile module
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
                file_content = extracted_file.read()
                file_like_object = BytesIO(file_content)
                nifti_image = nib.load(file_like_object)
        
        # Resize the image using the resize_image function
        resized_image = resize_image(nifti_image)
        
        # Convert the resized image to numpy array before returning
        image_data = resized_image.get_fdata()
        
        if self.transform:
            image_data = self.transform(image_data)
        
        return torch.tensor(image_data, dtype=torch.float32), os.path.basename(member.name)

def split_dataset(sftp, target_directory):
    split_ratio = 0.5, 0.2, 0.3 # train, val, holdout
    folder_names = ['training_images', 'validation_images', 'holdout_images']
    for folder in folder_names:
        folder_path = os.path.join(target_directory, 'resized_images', folder)
        sftp.mkdir(folder_path)
    torch.utils.data.random_split()


def resize_image(nifti_image):
    target_affine = np.eye(3)  # Identity matrix for affine transformation
    target_shape = (256, 256, 90)
    resampled_image = resample_img(nifti_image, target_affine=target_affine, target_shape=target_shape, interpolation='nearest')
    return resampled_image


def process_member(member, tar, sftp, target_directory):
    if member.name.startswith(target_directory) and not os.path.basename(member.name).startswith('._'):
        with tar.extractfile(member) as extracted_file:
            # Write the extracted file to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as temp_file:
                temp_file.write(extracted_file.read())
                temp_file_path = temp_file.name  # Save the path to the temporary file
            # Load the NIfTI image using nibabel
            nifti_image = nib.load(temp_file_path)
            # Resize the image
            resized_image = resize_image(nifti_image)
            # Save the resized image to a temporary file
            nib.save(resized_image, temp_file_path)
            # Create the resized_images directory on the remote server
            remote_resized_dir = os.path.join(target_directory, 'resized_images')
            sftp.mkdir(remote_resized_dir)
            # Upload the resized image to the resized_images directory
            remote_resized_path = os.path.join(remote_resized_dir, os.path.basename(member.name))
            sftp.put(temp_file_path, remote_resized_path)
            # Clean up the temporary files
            os.remove(temp_file_path)


if __name__ == "__main__":
    # Create SSH client
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Connect to the remote server
    client.connect(hostname='128.16.4.13', username='catalina', password='secret', timeout=30)

    # Use SFTP to open and read files
    sftp = client.open_sftp()

    # Remote tar path
    tar_path = '/raid/candi/catalina/Task03_Liver.tar'

    # Read the tar file remotely using SFTP
    with sftp.open(tar_path, 'rb') as remote_tar_file:

        # Open the tar file using the tarfile module
        with tarfile.open(fileobj=remote_tar_file, mode='r') as tar:
            target_directory = 'Task03_Liver/imagesTr/'
            # Use ThreadPoolExecutor to process files concurrently
            with ThreadPoolExecutor(max_workers=4) as executor:
                for member in tar.getmembers():
                    executor.submit(process_member, member, tar, sftp, target_directory)

    # Close the connection
    sftp.close()
    client.close()
