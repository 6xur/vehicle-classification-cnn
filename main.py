import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
    if(i == 3000):
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

# Hyperparameters
num_epochs = 20
batch_size = 4
learning_rate = 0.01

# Converting training data and labels to PyTorch Tensor
train_data = np.array(train_data)
train_data = torch.from_numpy(train_data)
vehicle_ids = torch.tensor(vehicle_ids, dtype=torch.long)

# Adjust labels to be in range 0 to 1361
if(torch.min(vehicle_ids).item() == 1):
    vehicle_ids = vehicle_ids - 1

# Create custom dataset from the tensors
train_dataset = CustomDataset(train_data, vehicle_ids)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1362)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ConvNet()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

model.train()
correct = 0
for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        # origin shape: [4, 1, 32, 64] = 4, 1, 2048
        #print(images.shape, labels)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        pred = outputs.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(labels.data.view_as(pred)).long().cpu().sum()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 200 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], step[{i + 1}], loss: {loss.item():.4f}')

    print(correct / len(train_loader.dataset))

