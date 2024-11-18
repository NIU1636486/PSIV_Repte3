# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:14:32 2024

Example of Main Steps for the Detection of HPilory using AutoEncoders for
the detection of anomalous pathological staining

Guides: 
    1. Split into train and test steps 
    2. Save trainned models and any intermediate result input of the next step
    
@authors: debora gil, pau cano
email: debora@cvc.uab.es, pcano@cvc.uab.es
Reference: https://arxiv.org/abs/2309.16053 


"""
# IO Libraries
import sys
import os
import pickle

# Standard Libraries
import numpy as np
import pandas as pd
import glob

# Torch Libraries
from torch.utils.data import DataLoader
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import wandb


## Own Functions
from Models.AEmodels import AutoEncoderCNN
from Models.datasets import Standard_Dataset
from loadCropped import loadCropped


wandb.login(key="8e9b2ed0a8b812e7888f16b3aa28491ba440d81a")
batch_size = 128
wandb.init(project="PSIV 3", config={"batch_size": 64, "nImages": 80}, dir="./wandb")


net_paramsEnc = {"drop_rate": 0}
net_paramsDec = {}
inputmodule_paramsDec = {}


def AEConfigs(Config):

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
        inputmodule_paramsDec["num_input_channels"] = net_paramsEnc["block_configs"][
            -1
        ][-1]

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
        inputmodule_paramsDec["num_input_channels"] = net_paramsEnc["block_configs"][
            -1
        ][-1]

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
        inputmodule_paramsDec["num_input_channels"] = net_paramsEnc["block_configs"][
            -1
        ][-1]

    return net_paramsEnc, net_paramsDec, inputmodule_paramsDec


######################### 0. EXPERIMENT PARAMETERS
# 0.1 AE PARAMETERS
inputmodule_paramsEnc = {}
inputmodule_paramsEnc["num_input_channels"] = 3

# 0.1 NETWORK TRAINING PARAMS
net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs("1")
criterion = nn.MSELoss()  # Reconstruction loss for AutoEncoder
num_epochs = 20
learning_rate = 1e-3
batch_size = 64

# 0.2 FOLDERS


print("Starting load Cropped...")
pathDir_servidor = "/fhome/maed/HelicoDataSet/CrossValidation/Cropped"


#### 1. LOAD DATA
# 1.1 Patient Diagnosis
x, y = loadCropped(os.listdir(pathDir_servidor), 100)
print("LOAD CROPPED DONE, filtrant")
images = list()
for image, diag in zip(x, y):
    if diag[2] == -1:
        images.append(image)

from torchvision import transforms

transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
transform = None
print("DADES FILTRADES")
dataset = Standard_Dataset(X=images, Y=images, transformation=transform)


# 1.2 Patches Data


#### 2. DATA SPLITING INTO INDEPENDENT SETS

# 2.0 Annotated set for FRed optimal threshold


# 2.1 AE trainnig set


# 2.1 Diagosis crossvalidation set

#### 3. lOAD PATCHES

### 4. AE TRAINING

# EXPERIMENTAL DESIGN:
# TRAIN ON AE PATIENTS AN AUTOENCODER, USE THE ANNOTATED PATIENTS TO SET THE
# THRESHOLD ON FRED, VALIDATE FRED FOR DIAGNOSIS ON A 10 FOLD SCHEME OF REMAINING
# CASES.

# 4.1 Data Split


dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
print("CREANT MODEL")
###### CONFIG1
Config = "1"
net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs(Config)
model = AutoEncoderCNN(
    inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print("MODEL CREAT")
# 4.2 Model Training
running_loss = 0.0
model.train()
for epoch in range(num_epochs):
    count = 0
    for inputs, _ in dataloader:
        inputs = inputs.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        count += 1
        print("COUNT:", count)
    epoch_loss = running_loss / len(dataloader)
    wandb.log({"loss": epoch_loss})
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")


model = model.to("cpu")
torch.save(model.state_dict(), "./model_rapid_nou.pth")


# Free GPU Memory After Training
torch.cuda.empty_cache()
gc.collect()
#### 5. AE RED METRICS THRESHOLD LEARNING

## 5.1 AE Model Evaluation

# Free GPU Memory After Evaluation
torch.cuda.empty_cache()
gc.collect()

## 5.2 RedMetrics Threshold

### 6. DIAGNOSIS CROSSVALIDATION
### 6.1 Load Patches 4 CrossValidation of Diagnosis

### 6.2 Diagnostic Power

wandb.finish()
