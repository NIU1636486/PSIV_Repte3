# -*- coding: utf-8 -*-
"""
Example of Main Steps for the Detection of HPilory using AutoEncoders
for the detection of anomalous pathological staining.

Guides:
1. Split into train and test steps
2. Save trained models and any intermediate result input of the next step
"""
# IO Libraries
import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import wandb

## Own Functions
from Models.AEmodels import AutoEncoderCNN
from Models.datasets import Standard_Dataset
from loadCropped import loadCropped

# WandB setup
wandb.login(key="8e9b2ed0a8b812e7888f16b3aa28491ba440d81a")
wandb.init(project="PSIV 3", config={"batch_size": 64, "nImages": 80}, dir="./wandb")

# Hyperparameters
batch_size = 64
learning_rate = 1e-3
num_epochs = 20

# AE Configurations
def AEConfigs(Config):
    net_paramsEnc = {"drop_rate": 0}
    net_paramsDec = {}
    inputmodule_paramsDec = {}

    if Config == "1":
        # CONFIG1
        net_paramsEnc["block_configs"] = [[32, 32], [64, 64]]
        net_paramsEnc["stride"] = [[1, 2], [1, 2]]
        net_paramsDec["block_configs"] = [
            [64, 32],
            [32, inputmodule_paramsEnc["num_input_channels"]],
        ]
        net_paramsDec["stride"] = net_paramsEnc["stride"]
        net_paramsEnc["drop_rate"] = 0.2
        net_paramsDec["drop_rate"] = 0.2
        inputmodule_paramsDec["num_input_channels"] = net_paramsEnc["block_configs"][-1][
            -1
        ]

    elif Config == "2":
        # CONFIG 2
        net_paramsEnc["block_configs"] = [[32], [64], [128], [256]]
        net_paramsEnc["stride"] = [[2], [2], [2], [2]]
        net_paramsDec["block_configs"] = [
            [128],
            [64],
            [32],
            [inputmodule_paramsEnc["num_input_channels"]],
        ]
        net_paramsDec["stride"] = net_paramsEnc["stride"]
        inputmodule_paramsDec["num_input_channels"] = net_paramsEnc["block_configs"][-1][
            -1
        ]

    elif Config == "3":
        # CONFIG3
        net_paramsEnc["block_configs"] = [[32], [64], [64]]
        net_paramsEnc["stride"] = [[1], [2], [2]]
        net_paramsDec["block_configs"] = [
            [64],
            [32],
            [inputmodule_paramsEnc["num_input_channels"]],
        ]
        net_paramsDec["stride"] = net_paramsEnc["stride"]
        inputmodule_paramsDec["num_input_channels"] = net_paramsEnc["block_configs"][-1][
            -1
        ]

    return net_paramsEnc, net_paramsDec, inputmodule_paramsDec


# Data Loading
print("Starting load Cropped...")
pathDir_servidor = "/fhome/maed/HelicoDataSet/CrossValidation/Cropped"
x, y = loadCropped(os.listdir(pathDir_servidor), 100)

# Filter images
print("Filtering data...")
images = [image for image, diag in zip(x, y) if diag[2] == -1]

# Transformations
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

# Split data into train and test
train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

# Create datasets
train_dataset = Standard_Dataset(X=train_images, Y=train_images, transformation=transform)
test_dataset = Standard_Dataset(X=test_images, Y=test_images, transformation=transform)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

# Model, Loss, Optimizer
print("Creating model...")
Config = "1"
inputmodule_paramsEnc = {"num_input_channels": 3}
net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs(Config)

model = AutoEncoderCNN(
    inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.MSELoss()  # Reconstruction loss for AutoEncoder
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training and Evaluation
print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, _ in train_dataloader:
        inputs = inputs.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_dataloader)

    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, _ in test_dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            test_loss += loss.item()

    test_loss /= len(test_dataloader)

    # Log results
    wandb.log({"Train Loss": train_loss, "Test Loss": test_loss})
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}"
    )

# Save model
print("Saving model...")
torch.save(model.state_dict(), "./model_hpilory_ae.pth")

# Free GPU Memory
torch.cuda.empty_cache()
gc.collect()

wandb.finish()
