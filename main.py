import numpy as np
import os
import torch
import torch.nn as nn

# Hyperparameters
num_epochs = 1
batch_size = 4
learning_rate = 0.001

train_directory = "vehicle-x/train"

class CustomDataset():
    def __init__(self, data_array, labels):
        self.data = data_array
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

train_data = []
vehicle_ids = []

# Load numpy arrays and labels from the training directory
for i, file_name in enumerate(os.listdir(train_directory)):

    # Limit elements in Tensor for testing, remove this when ready for full-scale training
    if(i == 12):
        break

    # Extract vehicle ID from the filename
    vehicle_id = int(file_name.split('_')[0])
    vehicle_ids.append(vehicle_id)

    # Load npy file
    file_path = os.path.join(train_directory, file_name)
    loaded_data = np.load(file_path)

    # Reshape loaded data and append to list
    # TODO: Need to check if this reshape is correct. What is the colour channel?
    reshaped_data = loaded_data.reshape((1, 32, 64))
    train_data.append(reshaped_data)

# Converting training data and labels to PyTorch Tensor
train_data = np.array(train_data)
train_data = torch.from_numpy(train_data)
vehicle_ids = torch.Tensor(vehicle_ids)

# Create custom dataset from the tensors
train_dataset = CustomDataset(train_data, vehicle_ids)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for i, (data,labels) in enumerate(train_loader):
        # origin shape: [4, 1, 32, 64] = 4, 1, 2048
        print(data.shape, labels)

