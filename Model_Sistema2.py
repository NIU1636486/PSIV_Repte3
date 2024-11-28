# IO Libraries
import sys
import os
import pickle

# Standard Libraries
import numpy as np
import pandas as pd
import glob
import torch.nn as nn
import torch.nn.functional as F

# Torch Libraries
from torch.utils.data import DataLoader
import gc
import torch

# import wandb
import torch.utils.data as data


from Models.AEmodels import AutoEncoderCNN
from Models.AEmodels import Encoder
from Models.AttentionUnits import GatedAttention
from loadCroppedMatrix import loadCropped
from torch.utils.data import DataLoader
import gc
import torch
import torch.nn as nn
import torch.optim as optim


# import wandb
# wandb.login(key="8e9b2ed0a8b812e7888f16b3aa28491ba440d81a")
# batch_size = 128
# wandb.init(project="PSIV 3", notes="ATTENTION", config={"batch_size": 1, "nImages": 100}, dir="./wandb")

class Standard_Dataset(data.Dataset):
    def __init__(self, X, Y=None, transformation=None):
        super().__init__()
        self.X = X
        self.y = Y
        self.transformation = transformation
 
    def __len__(self):
        
        return len(self.X)

    def __getitem__(self, idx):
        
        if self.y is not None:
            return torch.tensor(self.X[idx]).float(), torch.tensor(self.y[idx])
        else:
            return torch.from_numpy(self.X[idx])


net_paramsEnc = {"drop_rate": 0}
net_paramsDec = {}
inputmodule_paramsEnc={}
inputmodule_paramsEnc['num_input_channels']=3
inputmodule_paramsDec = {}
batch_size = 1
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

### CREAR ENCODER ####
Config = "1"
net_paramsEnc, _, _= AEConfigs(Config)
encoder_model = Encoder(
    inputmodule_paramsEnc, net_paramsEnc
)
# Carregar model preentrenat
encoder_model.load_state_dict(torch.load("model_encoder.pth"))
# Congelar l'encoder
for param in encoder_model.parameters():
    param.requires_grad = False


class AttentionModel(nn.Module):
    def __init__(self, encoder, attention_params, fc_params):
        super(AttentionModel, self).__init__()
        
        self.encoder = encoder
        self.attention = GatedAttention(attention_params)  # Your attention mechanism
        self.fc = nn.Sequential(
            nn.Linear(fc_params['in_features'], fc_params['hidden_units']),
            nn.ReLU(),
            nn.Dropout(fc_params['dropout']),
            nn.Linear(fc_params['hidden_units'], fc_params['out_features'])
        )

    def forward(self, x):
        x = x.squeeze(0)
        x = self.encoder(x)  
        pooled = F.adaptive_avg_pool2d(x, (1, 1))
        flattened = pooled.view(x.size(0), -1)  # Aplanar després del pooling
        z, attention_weights = self.attention(flattened)
        z = z.view(-1, self.attention.M*self.attention.ATTENTION_BRANCHES)
        x = self.fc(z)
        
        return x, attention_weights



attention_params = {
    'in_features': net_paramsEnc["block_configs"][-1][-1], 
    'decom_space': 16,  
    'ATTENTION_BRANCHES': 4  
}

fc_params = {
    'in_features': attention_params['ATTENTION_BRANCHES'] * attention_params['in_features'],  
    'hidden_units': 64,  
    'dropout': 0.5,  
    'out_features': 2
}
folder = "/fhome/maed/HelicoDataSet/CrossValidation/Cropped/"
folder = "./Cropped/"
print("Creant loadcropped")
x, y = loadCropped(os.listdir(folder), 10, "./PatientDiagnosis.csv")
print("loadcrppepd acabat")
dataset = Standard_Dataset(X=x, Y=y)


train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

learning_rate = 1e-3
num_epochs = 50
batch_size = 1
criterion = nn.CrossEntropyLoss()

attention_model = AttentionModel(encoder_model, attention_params, fc_params)
optimizer = optim.Adam(attention_model.parameters(), lr=learning_rate)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
attention_model.apply(initialize_weights)

attention_model.train()
loss_acum = 0
acum = 0
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        # Check input shape
        # Forward pass
        predictions, attention_weights = attention_model(inputs)
        
        # Compute loss
        loss_batch = criterion(predictions, labels)
        loss_acum += loss_batch
        acum += 1

        # Backward pass and optimization
       
        
        if acum == 10:
           # wandb_log = {"loss_batch": loss_acum / acum}
           # wandb.log(wandb_log)
            loss = loss_acum / acum
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_acum = 0
            acum = 0
            running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")
   # wandb.log({"loss_epoch": running_loss / len(train_loader)})

torch.save(attention_model.state_dict(), "model_attention.pth")

        