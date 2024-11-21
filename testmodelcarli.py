"""
@author: carlota
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
# import wandb



## Own Functions
from Models.AEmodels import AutoEncoderCNN
from Models.datasets import Standard_Dataset
#from loadCropped import loadCropped
from loadannotated import loadAnnotated
from reconstructionerror import *


# wandb.login(key="8e9b2ed0a8b812e7888f16b3aa28491ba440d81a")
# wandb.init(project="PSIV 3", config={"batch_size":512}, dir="./wandb")


net_paramsEnc = {"drop_rate": 0}
net_paramsDec = {}
inputmodule_paramsDec = {}
def AEConfigs(Config):

    if Config=='1':
        # CONFIG1
        net_paramsEnc['block_configs']=[[32,32],[64,64]]
        net_paramsEnc['stride']=[[1,2],[1,2]]
        net_paramsDec['block_configs']=[[64,32],[32,inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride']=net_paramsEnc['stride']
        net_paramsEnc["drop_rate"] = 0.2
        net_paramsDec["drop_rate"] = 0.2
        inputmodule_paramsDec['num_input_channels']=net_paramsEnc['block_configs'][-1][-1]

    elif Config=='2':
        # CONFIG 2
        net_paramsEnc['block_configs']=[[32],[64],[128],[256]]
        net_paramsEnc['stride']=[[2],[2],[2],[2]]
        net_paramsDec['block_configs']=[[128],[64],[32],[inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride']=net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels']=net_paramsEnc['block_configs'][-1][-1]

    elif Config=='3':
        # CONFIG3
        net_paramsEnc['block_configs']=[[32],[64],[64]]
        net_paramsEnc['stride']=[[1],[2],[2]]
        net_paramsDec['block_configs']=[[64],[32],[inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride']=net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels']=net_paramsEnc['block_configs'][-1][-1]

    return net_paramsEnc,net_paramsDec,inputmodule_paramsDec


######################### 0. EXPERIMENT PARAMETERS
# 0.1 AE PARAMETERS
inputmodule_paramsEnc={}
inputmodule_paramsEnc['num_input_channels']=3


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
print("CREANT MODEL")
###### CONFIG1
Config='1'
net_paramsEnc,net_paramsDec,inputmodule_paramsDec = AEConfigs(Config)
model=AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc,
                     inputmodule_paramsDec, net_paramsDec)


################# CARREGUEM MODEL #####################

import os
import torch
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.metrics import roc_curve


# Step 1: Load the pre-trained model
# Supongamos que AEConfigs y AutoEncoderCNN ya están definidos en tu entorno
net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs(Config)
model = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc,
                       inputmodule_paramsDec, net_paramsDec)
model.load_state_dict(torch.load('model_rapid_nou.pth'))
model.eval()  # Set the model to evaluation mode

# Step 2: Define transformations and initialize variables
# Define the transform (ToTensor will normalize the image)
transform = transforms.Compose([transforms.ToTensor()])
errors = []  # List to store reconstruction errors
labels = []  # List to store true labels



ListFolders = glob.glob("Annotated/*")
patientsImgs, patientsMeta = loadAnnotated(ListFolders, 50)



# Process each image and compute F_red
for input_image, metadata in zip(patientsImgs, patientsMeta):
    pat_id, img_window_id, label = metadata

    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    input_image = input_image.astype(np.float32) / 255.0  # Convert the image to float32 and normalize to [0, 1]

    # Convert image to a PIL Image (needed for transforms)
    input_image_pil = Image.fromarray((input_image * 255).astype(np.uint8))
    
    # Transform image to tensor
    input_tensor = transform(input_image_pil).unsqueeze(0)  # Add batch dimension

    # Pass the image through the model
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Postprocess the output image
    output_image = output_tensor.squeeze(0).detach().cpu().numpy()  # Remove batch dimension
    output_image = np.transpose(output_image, (1, 2, 0))  # Convert to HWC format

    # Convert images back to [0, 255] range\n")
    input_image_corrected = (input_image * 255).astype(np.uint8)
    output_image_corrected = (output_image * 255).astype(np.uint8)
    # Convert to BGR format as expected by OpenCV\n")
    input_image_bgr = cv2.cvtColor(input_image_corrected, cv2.COLOR_RGB2BGR)
    output_image_bgr = cv2.cvtColor(output_image_corrected, cv2.COLOR_RGB2BGR)
    
    # Compute F_red
    f_red = calculate_f_red(input_image_bgr, output_image_bgr, threshold_low=-20, threshold_high=20)
    print(f"F_red: {f_red}")

    # Collect errors and labels for ROC analysis
    errors.append(f_red)
    labels.append(label)  # Use the label extracted from the metadata

# Perform ROC analysis
optimal_threshold, fpr, tpr = roc_threshold_analysis(errors, labels)
print(f"Optimal Threshold: {optimal_threshold}")

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, label="ROC Curve")
plt.scatter([0], [1], color="red", label="Ideal Point")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

