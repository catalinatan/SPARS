import paramiko
import tarfile
import io
import nibabel as nib

# https://stackoverflow.com/questions/50457085/file-transfer-with-tar-piped-through-ssh-using-python
# https://medium.com/@keagileageek/paramiko-how-to-ssh-and-file-transfers-with-python-75766179de73

# Create SSH client
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# Connect to the remote server
client.connect(hostname='128.16.4.13', username='catalina', password='secret')

# Use SFTP to open and read files
sftp = client.open_sftp()

# Remote tar path
tar_path = '/raid/candi/catalina/Task03_Liver.tar'

# Read the tar file remotely using SFTP
with sftp.open(tar_path, 'rb') as remote_tar_file:
    # Open the tar file using the tarfile module
    with tarfile.open(fileobj=remote_tar_file, mode='r') as tar:
        target_directory = 'Task03_Liver/imagesTr/'
        for member in tar.getmembers():
            if member.name.startswith(target_directory):
                with tar.extractfile(member) as extracted_file:
                    # Load the NIfTI image directly from the file-like object
                    nifti_image = nib.load(extracted_file)
                    # Get the image dimensions (shape)
                    image_shape = nifti_image.shape
                    print(f"File: {member.name}, Dimensions: {image_shape}")
# Close the connection
sftp.close()
client.close()
