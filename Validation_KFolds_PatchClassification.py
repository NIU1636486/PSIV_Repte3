# -*- coding: utf-8 -*-

"""
Created on Tue Nov 19 12:59:58 2024

@author: marcp
"""

from Models.AEmodels import *

####################### MODEL ########################
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
# import wandb



## Own Functions
from Models.AEmodels import AutoEncoderCNN
from Models.datasets import Standard_Dataset
from loadCropped import loadCropped
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

import torch
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt

# Step 1: Load the pre-trained model
net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs(Config)
model = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc,
                       inputmodule_paramsDec, net_paramsDec)
model.load_state_dict(torch.load('model_rapid_nou.pth'))
model.eval()  # Set the model to evaluation mode

################ CARREGUEM LES DADES ###################
ListFolders = glob.glob("Cropped/*")
patientsImgsCropped, patientsMetaCropped = loadCropped(ListFolders,10)

ListFolders = glob.glob("Annotated/*")
patientsImgs, patientsMeta = loadAnnotated(ListFolders, 10)

################ CARREGUEM EL MODEL ##############
import os
import torch
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
K = 10 #Number of folds

# Step 1: Load the pre-trained model
# Supongamos que AEConfigs y AutoEncoderCNN ya están definidos en tu entorno
net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs(Config)
model = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc,
                    inputmodule_paramsDec, net_paramsDec)
model.load_state_dict(torch.load('model_rapid_nou.pth'))
model.eval()  # Set the model to evaluation mode


print("\nStarting with Patch classification...")
folds_optimals_tresholds = list()
patientsID = [x[0] for x in patientsMeta]
y = [x[2] for x in patientsMeta]
X = range(len(patientsImgs))
kf = GroupKFold(n_splits=2)
folds = list()

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y, patientsID)):
    print(f'\nProcessant Fold {fold}...')
    
    # Step 2: Define transformations and initialize variables
    transform = transforms.Compose([transforms.ToTensor()])
    errors = []  # List to store reconstruction errors
    labels = []  # List to store true labels
    
    # Recuperem les imatges train
    train_imgs = [patientsImgs[i] for i in train_idx]
    # Recuperem les imatges test
    test_imgs = [patientsImgs[i] for i in val_idx]
    # Recuperem la label de les imatges train
    train_label = [y[i] for i in train_idx]
    # Recuperem els patients utilitzats pel train del fold
    train_patients = list(set([patientsMeta[i][0] for i in train_idx]))
    # Recuperem els patients utilitzats pel test del fold
    test_patients = list(set([patientsMeta[i][0] for i in val_idx]))
    # Recuperem les metadades del conjunt de validació
    test_meta = [patientsMeta[i] for i in val_idx]
    # Afegim a la llista de folds els pacients utilitzats
    folds.append([train_patients,test_patients])
    
    # Train reconstruction error
    print(f'\nTraining Fold {fold}....')
    for input_image, label in zip(train_imgs,train_label):
        
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.astype(np.float32) / 255.0
        
        input_tensor = transform(input_image).unsqueeze(0)
        
        # Pass the image through the model
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        # Postprocess the output image
        output_image = output_tensor.squeeze(0).detach().cpu().numpy()  # Remove batch dimension
        output_image = np.transpose(output_image, (1, 2, 0))  # Convert to HWC format
        
        # Compute reconstruction error and F_red
        reconstruction_error = compute_reconstruction_error(input_image, output_image)
        f_red = calculate_f_red(input_image, output_image)
        
        # Print results for the current image
        #print(f"Image: {img}")
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
    folds_optimals_tresholds.append([optimal_threshold, fpr, tpr])
    
    

    print(f'\nValidation Fold {fold}....')
    patch_classification = []
    for input_image, meta in zip(test_imgs,test_meta):
        patID, nameImg, label = meta
        
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.astype(np.float32) / 255.0
        
        input_tensor = transform(input_image).unsqueeze(0)
        
        # Pass the image through the model
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        # Postprocess the output image
        output_image = output_tensor.squeeze(0).detach().cpu().numpy()  # Remove batch dimension
        output_image = np.transpose(output_image, (1, 2, 0))  # Convert to HWC format
        
        # Compute reconstruction error and F_red
        reconstruction_error = compute_reconstruction_error(input_image, output_image)
        f_red = calculate_f_red(input_image, output_image)
        
        if f_red >= optimal_threshold:
            patch_classification.append([1,label])
        
        else:
            patch_classification.append([-1,label])

print("\nStarting with Patient Diagnosis....")

folds_optimals_diagnosis = list()
diagnosis = {}
for fold, (train_pat, val_pat) in enumerate(folds):
    patientsDiagnosisImgs_train = list()
    patientsDiagnosisMeta_train = list()
    patientsDiagnosisImgs_test = list()
    patientsDiagnosisMeta_test = list()

    
    for x,y in zip(patientsImgsCropped, patientsMetaCropped):
        
        if y[0] in train_pat:
            patientsDiagnosisImgs_train.append(x)
            patientsDiagnosisMeta_train.append(y)
            if y[0] not in diagnosis:
                diagnosis[y[0]] = y[2]
        
        elif y[0] in val_pat:
            patientsDiagnosisImgs_test.append(x)
            patientsDiagnosisMeta_test.append(y)
    errors = list()
    labels = list()
    positive = {}
    print(f"\nTraining fold {fold}...")
    optimalTreshold = folds_optimals_tresholds[fold]
    for input_image, meta in zip(patientsDiagnosisImgs_train,patientsDiagnosisMeta_train):      
         
        patID, nameImg, label = meta
       
        input_tensor = transform(input_image).unsqueeze(0)
        
        # Pass the image through the model
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        # Postprocess the output image
        output_image = output_tensor.squeeze(0).detach().cpu().numpy()  # Remove batch dimension
        output_image = np.transpose(output_image, (1, 2, 0))  # Convert to HWC format
        
        # Compute reconstruction error and F_red
        reconstruction_error = compute_reconstruction_error(input_image, output_image)
        f_red = calculate_f_red(input_image, output_image)
        

        if patID not in positive:
            positive[patID] = 0
            
        if f_red > optimal_threshold:
            positive[patID] += 1
        

    for key,value in positive.items():
        diag = (value/10)
        errors.append(diag)
        labels.append(diagnosis[key])
        print(f"Percentatge de finestres positives pel pacient {key} =",diag*100,"%")
    
    
    # Perform ROC analysis
    optimal_diagnosis, fpr, tpr = roc_threshold_analysis(errors, labels)
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
    folds_optimals_diagnosis.append([optimal_diagnosis, fpr, tpr])
    
    print(f'\nValidation Fold {fold}....')
    patient_diagnosis = []
    positive = {}
    for input_image, meta in zip(patientsDiagnosisImgs_test, patientsDiagnosisMeta_test):
        patID, nameImg, label = meta
        
            
        input_tensor = transform(input_image).unsqueeze(0)
        
        # Pass the image through the model
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        # Postprocess the output image
        output_image = output_tensor.squeeze(0).detach().cpu().numpy()  # Remove batch dimension
        output_image = np.transpose(output_image, (1, 2, 0))  # Convert to HWC format
        
        # Compute reconstruction error and F_red
        reconstruction_error = compute_reconstruction_error(input_image, output_image)
        f_red = calculate_f_red(input_image, output_image)
        

        if patID not in positive:
            positive[patID] = 0
            
        if f_red > optimal_threshold:
            positive[patID] += 1
        
    for key,value in positive.items():
        diag = (value/10)
        print(f"Percentatge de finestres positives pel pacient {key} =",diag*100,"%")
        if diag >= optimal_diagnosis:
            patient_diagnosis.append([1,diagnosis[key]])
            print(f"El pacient {key} té la presencia de l'helicobacter")
        
        else:
            patient_diagnosis.append([-1,diagnosis[key]])
            print(f"El pacient {key} no té la presencia de l'helicobacter")
            
