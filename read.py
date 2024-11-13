
import paramiko
import tarfile
import torch
from torch.utils.data import Dataset, random_split
import nibabel as nib

# https://stackoverflow.com/questions/50457085/file-transfer-with-tar-piped-through-ssh-using-python

# Create SSH client
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# Connect to the remote server
client.connect(hostname='128.16.4.13', username='catalina', password='secret')

# Use SFTP to open and read files
sftp = client.open_sftp()

# Specify the path to the parent directory 
parent_directory = '/raid/candi/catalina/Task03_Liver.tar/imagesTr'

# placeholder for splitting the imagesTr into 3 folders
# Close the connection
sftp.close()
client.close()
