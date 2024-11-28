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
batch_size = 1
# wandb.init(project="PSIV 3", notes="ATTENTION", config={"batch_size": 1, "nImages": 100}, dir="./wandb")

embeddings = np.load("patient_embeddings.npy")
labels = np.load("patient_labels.npy")

# Define a Dataset for embeddings
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx]).float(), torch.tensor(self.labels[idx])


embedding_dataset = EmbeddingDataset(embeddings, labels)
embedding_loader = DataLoader(embedding_dataset, batch_size=batch_size, shuffle=True)

epochs = 50
# Define training parameters
attention_model = AttentionModel(None, attention_params, fc_params)  # Pass None as encoder isn't used
optimizer = optim.Adam(attention_model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


class AttentionModel(nn.Module):
    def __init__(self, attention_params, fc_params):
        super(AttentionModel, self).__init__()
        
        self.attention = GatedAttention(attention_params)  # Your attention mechanism
        self.fc = nn.Sequential(
            nn.Linear(fc_params['in_features'], fc_params['hidden_units']),
            nn.ReLU(),
            nn.Dropout(fc_params['dropout']),
            nn.Linear(fc_params['hidden_units'], fc_params['out_features'])
        )

    def forward(self, x):
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



learning_rate = 1e-3
num_epochs = 50
batch_size = 1
criterion = nn.CrossEntropyLoss()

attention_model = AttentionModel(attention_params, fc_params)
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
       
        
        if acum == 77:
            loss = loss_acum / acum
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_acum = 0
            acum = 0
            running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")
   # wandb.log({"loss_epoch": running_loss / len(train_loader)})

torch.save(attention_model.state_dict(), "model_attention_dos.pth")

        