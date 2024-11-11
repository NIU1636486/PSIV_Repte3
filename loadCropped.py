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
    patientsImgs = np.array(0)
    excel = pd.read_csv(pathExcel)
    for pathDir in ListFolders:
        imgs = glob.glob(os.path.join(pathDir,'*.png'))
        n = len(imgs)
        for _ in range(nImages):
            index = random.randint(0,n)
            img = cv2.imread(imgs[index])
            diagnosis = excel.loc[excel['CODI'] == pathDir, 'DENSITAT'].values[0]

            if diagnosis == "NEGATIVA":
                diagnosis = 0
            elif diagnosis == "BAIXA" or diagnosis == "ALTA":
                diagnosis = 1
            metadates = [pathDir,imgs[index],diagnosis]
            patientsImgs.append([img,metadates])
    
    return patientsImgs

            
    
    