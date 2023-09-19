import numpy as np
import os
import torch
import torch.nn as nn

# Hyper-parameters
num_epochs = 1
batch_size = 4
learning_rate = 0.001

train_directory = "vehicle-x/train"

train_list = []
vehicle_ids = []


class CustomDataset():
    def __init__(self, data_array, labels):
        self.data = data_array
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

i = 0
# Load all npy files into a single Numpy array
for file_name in os.listdir(train_directory):

    # Limit elements in Tensor for testing, to be removed
    if(i == 12):
        break
    i = i + 1

    # Extract vehicle ID from the filename
    vehicle_id = int(file_name.split('_')[0])
    vehicle_ids.append(vehicle_id)

    # Load npy file and append to list
    file_path = os.path.join(train_directory, file_name)
    loaded_data = np.load(file_path)
    # TODO: Need to check if this reshape is correct. What is the colour channel?
    reshaped_data = loaded_data.reshape((1, 32, 64))
    train_list.append(reshaped_data)

train_array = np.array(train_list)
train_tensor = torch.from_numpy(train_array)

vehicle_ids = torch.Tensor(vehicle_ids)

train_dataset = CustomDataset(train_tensor, vehicle_ids)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for i, (data,labels) in enumerate(train_loader):
        # origin shape: [4, 1, 32, 64] = 4, 1, 2048
        print(data.shape, labels)

