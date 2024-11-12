# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:43:05 2024

@author: marcp
"""
import glob
import os
import numpy as np
import random
import cv2
import pandas as pd
def loadCropped(ListFolders, nImages):
    pathExcel = "PatientDiagnosis.csv"
    patientsImgs = list()
    patientsMeta= []
    excel = pd.read_csv(pathExcel)
    for pathDir in ListFolders:
        imgs = glob.glob(os.path.join(pathDir,'*.png'))
        n = len(imgs)
        for _ in range(nImages):
            index = random.randint(0,n-1)
            img = cv2.imread(imgs[index])
            diagnosis = excel.loc[excel['CODI'] == pathDir[:-2], 'DENSITAT'].values[0]

            if diagnosis == "NEGATIVA":
                diagnosis = -1
            elif diagnosis == "BAIXA" or diagnosis == "ALTA":
                diagnosis = 1
            patientsMeta.append([pathDir[:-2],imgs[index],diagnosis])
            patientsImgs.append(img)
    
    return patientsImgs, patientsMeta


# Directori Actual
directorio_actual = os.getcwd()

# Llistar les carpetes del directori actual
carpetas = [nombre for nombre in os.listdir(directorio_actual) if os.path.isdir(os.path.join(directorio_actual, nombre))]

patientsImgs, patientsMeta = loadCropped(carpetas,20)

    
    