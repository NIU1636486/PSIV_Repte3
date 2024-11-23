# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 11:14:27 2024

@author: marcp
"""

########################### LLIBRERIES ##################################

# Importem llibreries io
import sys
import os
import pickle


# Importem les llibreries relacionades amb Pytorch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch
import gc

# Importem llibreries per la validació
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import roc_curve


# Importem llibreries pel processament d'imatges
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import glob
import cv2

########################### FUNCIONS ####################################

# Importem les funcions propies necessaries
from Models.datasets import Standard_Dataset
from loadannotated import loadAnnotated
from loadCropped import loadCropped
from reconstructionerror import *
from Models.AEmodels import *


##################### CONFIGURACIONS AUTOENCODER ########################

net_paramsEnc = {"drop_rate": 0}
net_paramsDec = {}
inputmodule_paramsDec = {}



def AEConfigs(Config):

    if Config == '1':
        # CONFIG1
        net_paramsEnc['block_configs'] = [[32, 32], [64, 64]]
        net_paramsEnc['stride'] = [[1, 2], [1, 2]]
        net_paramsDec['block_configs'] = [[64, 32], [
            32, inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride'] = net_paramsEnc['stride']
        net_paramsEnc["drop_rate"] = 0.2
        net_paramsDec["drop_rate"] = 0.2
        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]

    elif Config == '2':
        # CONFIG 2
        net_paramsEnc['block_configs'] = [[32], [64], [128], [256]]
        net_paramsEnc['stride'] = [[2], [2], [2], [2]]
        net_paramsDec['block_configs'] = [[128], [64], [32],
                                          [inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride'] = net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]

    elif Config == '3':
        # CONFIG3
        net_paramsEnc['block_configs'] = [[32], [64], [64]]
        net_paramsEnc['stride'] = [[1], [2], [2]]
        net_paramsDec['block_configs'] = [[64], [32], [
            inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride'] = net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels'] = net_paramsEnc['block_configs'][-1][-1]

    return net_paramsEnc, net_paramsDec, inputmodule_paramsDec

###################### EXPERIMENT PARAMETRES AE #########################


inputmodule_paramsEnc = {}
inputmodule_paramsEnc['num_input_channels'] = 3

########################### CREACIÓ MODEL ####################################

print("CREANT MODEL")
# CONFIG1
Config = '1'
net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs(Config)
model = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc,
                       inputmodule_paramsDec, net_paramsDec)

########################## CARREGUEM MODEL #################################

# Step 1: Load the pre-trained model
net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs(Config)
model = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc,
                       inputmodule_paramsDec, net_paramsDec)
model.load_state_dict(torch.load('model_rapid_nou_v2.pth'))
model.eval()  # Set the model to evaluation mode

########################## CARREGUEM DADES ################################

# Dades per la validació del patitent diagnosis
ListFolders = glob.glob("Cropped/*")
patientsImgsCropped, patientsMetaCropped = loadCropped(ListFolders, 50)

# Dades per la validació del patch classification
ListFolders = glob.glob("Annotated/*")
patientsImgsAnnotated, patientsMetaAnnotated = loadAnnotated(ListFolders, 50)

######################## PREPARACIÓ DADES VALIDACIÓ ######################

print("\nPreparació de les dades i estructures per la validació en Kfolds...")

# Llistes per guardar els thresholds optims dels folds
OptimalThresholdsPatchClassification = list()
OptimalThresholdsPatientsDiagnosis = list()

# LListes per guardar les mètriques de validació
PrecisionPatchClassification = list()
RecallPatchClassification = list()
F1ScorePatchClassification = list()
ConfusionMatrixPatchClassification = list()
PrecisionPatientsDiagnosis = list()
RecallPatientsDiagnosis = list()
F1ScorePatientsDiagnosis = list()
ConfusionMatrixPatientsDiagnosis = list()

# Preparem la configuració dels folds
kf = GroupKFold(n_splits=2)

# Llista per guardar les particions de pacients train/test dels 10 folds
patientsFolds = list()

# Preparem les dades per la partició
X = range(len(patientsImgsAnnotated))
y = [x[2] for x in patientsMetaAnnotated]
patientsID = [y[0] for y in patientsMetaAnnotated]

print("\nStarting with Patch classification...")

################### PREPARACIÓ FOLDS PATCH CLASSIFICATION #####################

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y, patientsID)):
    
    print(f'\nProcessant Fold {fold}...')
    
    # Recuperem les imatges del train pel fold actual
    trainImgs = [patientsImgsAnnotated[i] for i in train_idx]
    
    # Recuperem les metadades del train pel fold actual
    trainMeta = [patientsMetaAnnotated[i] for i in train_idx]
    
    # Recuperem les imatges del test pel fold actual
    testImgs = [patientsImgsAnnotated[i] for i in test_idx]
    
    # Recuperem les metadades del test pel fold actual
    testMeta = [patientsMetaAnnotated[i] for i in test_idx]
    
    # Recuperem els patients utilitzats pel train/test del fold actual
    trainPatients = list(set([patientsMetaAnnotated[i][0] for i in train_idx]))
    testPatients = list(set([patientsMetaAnnotated[i][0] for i in test_idx]))
    
    # Guardem la particio de pacients test/train del fold actual
    patientsFolds.append([trainPatients, testPatients])
    
    
    # Definim les transformacions i variables per l'entrenament
    transform = transforms.Compose([transforms.ToTensor()])
    errors = []  # List to store reconstruction errors
    labels = []  # List to store true labels
    
####################### TRAIN PATCH CLASSIFICATION ###########################

    print(f'\nTraining Patch classification Fold {fold}....')
    
    for input_image, meta in zip(trainImgs,trainMeta):
        
        # Extraiem les metadades
        patID, imgName, label = meta
        
        # Preparem les dades per passar-les per l'autoencoder
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # Convertim la imatge a RGB
        input_image = input_image.astype(np.float32) / 255.0  # Convertim img a float32 i normalitzem en [0, 1]
        input_image_pil = Image.fromarray((input_image * 255).astype(np.uint8)) # Convertim a PIL Image (necessari per transformacions)
        input_tensor = transform(input_image_pil).unsqueeze(0)  # Transformem imatge a tensor
        
        # Passem el tensor pel model autoencoder
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # Processament de la sortida de l'autoencoder
        output_image = output_tensor.squeeze(0).detach().cpu().numpy()  # Eliminem la dimensió de batch
        output_image = np.transpose(output_image, (1, 2, 0))  # Convertim la imatge en format HWC
        input_image_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR) # Convertim la img d'entrada a BGR
        output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR) # Convertim la img de sortida a BGR
        
        # Calculem la mesura f_red
        f_red = calculate_f_red(input_image_bgr, output_image_bgr, threshold_low=-20, threshold_high=20)
        #print(f"F_red: {f_red}")
        
        # Recollim les mesures d'error i l'etiqueta real de la imatge 
        errors.append(f_red)
        labels.append(label)  
    
    
    # Calculem el threshold òptim del fold actual pel patch classification
    optimalThreshold, fpr, tpr = roc_threshold_analysis(errors, labels)
    
    # Guardem el threshold òptim del fold actual amb els altres
    OptimalThresholdsPatchClassification.append(optimalThreshold)
    
    # Mostrem la corba ROC de l'analisi de l'entrenament del fold actual
    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve (Patch Classification")
    plt.scatter([0], [1], color="red", label="Ideal Point")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Patch Classification")
    plt.legend()
    plt.show()
    
    print(f"Optimal Threshold: {optimalThreshold}")
    
######################### TEST PATCH CLASSIFICATION ###########################

    print(f'\nTest Patch classification Fold {fold}....')
    
    # Llista per guardar les etiquetes reals i prediccions de la classificació
    patchClassification = list()
    
    for input_image, meta in zip(testImgs,testMeta):
        
        # Recuperem les metadades de la imatge actual
        patID, nameImg, label = meta
        
        # Processament de la imatge d'entrada al model
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # Convertim la img en format RGB
        input_image = input_image.astype(np.float32) / 255.0  # Convertim la img a float32 i normalitzar a [0, 1]
        input_image_pil = Image.fromarray((input_image * 255).astype(np.uint8)) # Convertim la img a PIL per la transformació
        input_tensor = transform(input_image_pil).unsqueeze(0)  # Transformem la imatge en tensor 
    
        # Passem la imatge pel model AE
        with torch.no_grad():
            output_tensor = model(input_tensor)
            
        # Processament de la imatge de sortida
        output_image = output_tensor.squeeze(0).detach().cpu().numpy()  # Eliminem la dimensió batch
        output_image = np.transpose(output_image, (1, 2, 0))  # Convertim la img a format HWC
        input_image_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR) # Convertim la img d'entrada a BGR
        output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)# Convertim la img de sortida a BGR
    
        # Calculem la mesura f_red de la reconstrucció de la imatge
        f_red = calculate_f_red(input_image_bgr, output_image_bgr, threshold_low=-20, threshold_high=20)
        
  
        # Classifiquem el patch segons el optimalThreshold del fold actual calculat 
        if f_red >= optimalThreshold:
            patchClassification.append([label,1])
        
        else:
            patchClassification.append([label,-1])
   
############################# METRIQUES DE VALIDACIÓ #########################################################    

    # Dividim la classificació en reals i predites
    realLabels = [item[0] for item in patchClassification]
    predLabels = [item[1] for item in patchClassification]

    # Calculem les metriques de validació

    precision = precision_score(realLabels, predLabels, pos_label= 1)  
    recall = recall_score(realLabels, predLabels, pos_label= 1)
    f1 = f1_score(realLabels, predLabels, pos_label = 1)
    confusionMatrix = confusion_matrix(realLabels, predLabels)
    
    # Guardem les metriques de validació
    
    PrecisionPatchClassification.append(precision)
    RecallPatchClassification.append(recall)
    F1ScorePatchClassification.append(f1)
    ConfusionMatrixPatchClassification.append(confusionMatrix)
    
    # Imprimir resultados
    print(f"Precisió (Precision): {precision}")
    print(f"Sensibilitat (Recall): {recall}")
    print(f"F1-score: {f1}")
    print("Matriu de Confusió:")
    print(confusionMatrix)
    
    # Reporte detallado
    print("\nReporte de clasificación:")
    print(classification_report(realLabels, predLabels, labels=[1, -1]))

print("\nStarting with Patients Diagnosis...")

################### PREPARACIÓ FOLDS PATIENT DIAGNOSIS #####################

# Diccionari per guardar els diagnostics reals dels pacients
patientsRealDiagnosis = {}

# Formem els conjunts test/train per cada fold
for fold, (trainPatients, testPatients) in enumerate(patientsFolds):
    
    print(f'\nProcessant Fold {fold}...')
    
    # Inicialitzem la llista d'imatges de train
    trainImgs = list()
    # Inicialitzem la llista de metadades de train
    trainMeta = list()
    # Inicialitzem la llista d'imatges de test
    testImgs = list()
    # Inicialitzem la llista de metadades de test
    testMeta = list()
    
    # Recorrem les imatges i metadades cropped
    for x,y in zip(patientsImgsCropped, patientsMetaCropped):
        
        # Afegim les dades dels pacients de train pel fold actual
        if y[0] in trainPatients:
            trainImgs.append(x)
            trainMeta.append(y)
            if y[0] not in patientsRealDiagnosis:
                patientsRealDiagnosis[y[0]] = y[2]
        
        # Afegim les dades dels pacients de test pel fold actual
        elif y[0] in testPatients:
            testImgs.append(x)
            testMeta.append(y)
            if y[0] not in patientsRealDiagnosis:
                patientsRealDiagnosis[y[0]] = y[2]
    
    # Definim les transformacions i variables per l'entrenament
    transform = transforms.Compose([transforms.ToTensor()])
    errors = []  # Llista per guardar els errors de reconstrucci´`o
    labels = []  # Llista per guardar labels reals
    positivesDiagnosis = {} # Diccionari per acumular patches positius
    
####################### TRAIN PATIENT DIAGNOSIS ###############################

    print(f'\nTraining Patient diagnosis Fold {fold}....')

    # Recuperem el optimal threshold pel fold actual
    optimalTreshold = OptimalThresholdsPatchClassification[fold]
    
    for input_image, meta in zip(trainImgs,trainMeta):      
         
        # Extraiem les metadades
        patID, nameImg, label = meta
        
        # Processem la imatge per passar-la per l'autoencoder
        input_image = np.transpose(input_image, (1, 2, 0)) # Transposem la imatge
        input_image_pil = Image.fromarray((input_image * 255).astype(np.uint8)) # Convertim la img a PIL per la transformació
        input_tensor = transform(input_image_pil).unsqueeze(0)  # Transformem la imatge en tensor 
        
        # Passem la imatge pel model AE
        with torch.no_grad():
            output_tensor = model(input_tensor)
            
        # Processament de la imatge de sortida
        output_image = output_tensor.squeeze(0).detach().cpu().numpy()  # Eliminem la dimensió batch
        output_image = np.transpose(output_image, (1, 2, 0))  # Convertim la img a format HWC
        input_image_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR) # Convertim la img d'entrada a BGR
        output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)# Convertim la img de sortida a BGR
        
        # Calculem la mesura d'error f_red 
        f_red = calculate_f_red(input_image_bgr, output_image_bgr, threshold_low=-20, threshold_high=20)
              
        # Comprovem si el pacient es dins el diccionari
        if patID not in positivesDiagnosis:
            positivesDiagnosis[patID] = [0,0]
        
        # Contem els patches dels pacients que estem entrenant
        positivesDiagnosis[patID][1] += 1
        
        # Comprovem si el patch del pacient supera el threshold optim
        if f_red > optimalThreshold:
            positivesDiagnosis[patID][0] += 1
        
    # Calculem el percentatge de finestres positives dels pacients
    for key,value in positivesDiagnosis.items():
        res = (value[0]/value[1])
        errors.append(res)
        labels.append(patientsRealDiagnosis[key])
        print(f"Percentatge de finestres positives pel pacient {key} =",res*100,"%")
    
    
    # Calculem el threshold òptim per el patient diagnosis
    optimalDiagnosis, fpr, tpr = roc_threshold_analysis(errors, labels)
    print(f"Optimal Threshold: {optimalDiagnosis}")
    
    # Afegim el threshold òptim amb la resta 
    OptimalThresholdsPatientsDiagnosis.append(optimalDiagnosis)
    
    # Mostrem la ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve (Patient Diagnosis)")
    plt.scatter([0], [1], color="red", label="Ideal Point")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
   
    print(f'\nTest Patient Diagnosis Fold {fold}....')
    
    # Llista per guardar els diagnostics del pacient reals i predits
    patientDiagnosis = []
    
    # Diccionari per comptar els diagnostics positius
    positivesDiagnosis = {}
    
    for input_image, meta in zip(testImgs, testMeta):
        
        # Recuperem les metadades de la imatge
        patID, nameImg, label = meta
        
        # Processem la imatge per passar-la per l'autoencoder
        input_image = np.transpose(input_image, (1, 2, 0)) # Transposem la imatge
        input_image_pil = Image.fromarray((input_image * 255).astype(np.uint8)) # Convertim la img a PIL per la transformació
        input_tensor = transform(input_image_pil).unsqueeze(0)  # Transformem la imatge en tensor 
          
        
        # Passem la imatge pel model AE
        with torch.no_grad():
            output_tensor = model(input_tensor)
            
        # Processament de la imatge de sortida
        output_image = output_tensor.squeeze(0).detach().cpu().numpy()  # Eliminem la dimensió batch
        output_image = np.transpose(output_image, (1, 2, 0))  # Convertim la img a format HWC
        input_image_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR) # Convertim la img d'entrada a BGR
        output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)# Convertim la img de sortida a BGR
        
        # Calculem la mesura d'error f_red 
        f_red = calculate_f_red(input_image_bgr, output_image_bgr, threshold_low=-20, threshold_high=20)
        
        #Comprovem si el pacient es dins el diccionari
        if patID not in positivesDiagnosis:
            positivesDiagnosis[patID] = [0,0]
       
        # Contem els patches dels pacients que estem validant
        positivesDiagnosis[patID][1] += 1
       
        # Comprovem si el patch del pacient supera el threshold optim
        if f_red > optimalThreshold:
           positivesDiagnosis[patID][0] += 1
           
    # Calculem el percentatge de finestres positives dels pacients
    for key, value in positivesDiagnosis.items():
        res = (value[0]/value[1])
        print(f"Percentatge de finestres positives pel pacient {key} =",res*100,"%")
        if res >= optimalDiagnosis:
            patientDiagnosis.append([patientsRealDiagnosis[key],1])
        
        else:
            patientDiagnosis.append([patientsRealDiagnosis[key],-1])
            print(f"El pacient {key} no té la presencia de l'helicobacter")
            
      
############################# METRIQUES DE VALIDACIÓ #########################################################    

    # Dividim la classificació en reals i predites
    realLabels = [item[0] for item in patientDiagnosis]
    predLabels = [item[1] for item in patientDiagnosis]

    # Calculem les metriques de validació

    precision = precision_score(realLabels, predLabels, pos_label= 1)  
    recall = recall_score(realLabels, predLabels, pos_label= 1)
    f1 = f1_score(realLabels, predLabels, pos_label = 1)
    confusionMatrix = confusion_matrix(realLabels, predLabels)
    
    # Guardem les metriques de validació
    
    PrecisionPatientsDiagnosis.append(precision)
    RecallPatientsDiagnosis.append(recall)
    F1ScorePatientsDiagnosis.append(f1)
    ConfusionMatrixPatientsDiagnosis.append(confusionMatrix)
    
    # Imprimir resultados
    print(f"Precisió (Precision): {precision}")
    print(f"Sensibilitat (Recall): {recall}")
    print(f"F1-score: {f1}")
    print("Matriu de Confusió:")
    print(confusionMatrix)
    
    # Reporte detallado
    print("\nReporte de clasificación:")
    print(classification_report(realLabels, predLabels, labels=[1, -1]))
            
    
    
############################# ESTADISITIQUES DELS THRESHOLDS ##################################
# Calculem estadístiques pels thresholds del patch classification
mean_threshold = np.mean(OptimalThresholdsPatchClassification)  # m
std_threshold = np.std(OptimalThresholdsPatchClassification)   # Desviación estándar
cv_threshold = std_threshold / mean_threshold  # Coeficiente de variación

# Mostrar resultados
print("\n ESTADÍSTIQUES THRESHOLDS PATCH CLASSIFICATION \n")
print(f"Media de thresholds (μ): {mean_threshold:.4f}")
print(f"Desviación estándar de thresholds (σ): {std_threshold:.4f}")
print(f"Coeficiente de variación (CV): {cv_threshold:.4%}")


plt.boxplot(OptimalThresholdsPatchClassification, vert=False)
plt.title('Dispersión de Thresholds Patch Classification')
plt.xlabel('Threshold')
plt.show()

# Calculem estadístiques pels thresholds del patient diagnosis
mean_threshold = np.mean(OptimalThresholdsPatientsDiagnosis)  # m
std_threshold = np.std(OptimalThresholdsPatientsDiagnosis)   # Desviación estándar
cv_threshold = std_threshold / mean_threshold  # Coeficiente de variación

# Mostrar resultados
print("\n ESTADÍSTIQUES THRESHOLDS PATIENT DIAGNOSIS \n")
print(f"Media de thresholds (μ): {mean_threshold:.4f}")
print(f"Desviación estándar de thresholds (σ): {std_threshold:.4f}")
print(f"Coeficiente de variación (CV): {cv_threshold:.4%}")


plt.boxplot(OptimalThresholdsPatientsDiagnosis, vert=False)
plt.title('Dispersión de Thresholds Patient Diagnosis')
plt.xlabel('Threshold')
plt.show()
