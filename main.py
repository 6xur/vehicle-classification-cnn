import numpy as np
import os
import torch

train_directory = "vehicle-x/train"

train_list = []

i = 0
# Load all npy files into a single Numpy array
for file_name in os.listdir(train_directory):

    # Limit elements in Tensor for testing, to be removed
    if(i == 5):
        break
    i = i + 1

    file_path = os.path.join(train_directory, file_name)
    loaded_data = np.load(file_path)
    train_list.append(loaded_data)

train_array = np.array(train_list)

train_tensor = torch.from_numpy(train_array)
print(f'Length of the tensor is {train_tensor.shape}')
print(f'Length of the first element in the tensor is {train_tensor[0].shape}')
