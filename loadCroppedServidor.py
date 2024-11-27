# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 18:58:27 2024

@author: marcp
"""
import glob
import os
import numpy as np
import random
import cv2
import pandas as pd
import skimage.io as io

def loadCropped(ListFolders, nImgs):
    pathExcel = "/fhome/maed/HelicoDataSet/PatientDiagnosis.csv"
    patientsImgs = list()
    patientsMeta= []
    excel = pd.read_csv(pathExcel)
    nImages = nImgs

    for pathDir in ListFolders:
        print("Loading images from", pathDir)
        pathdir_llarg = os.path.join("/fhome/maed/HelicoDataSet/CrossValidation/Cropped/", pathDir)
        imgs = os.listdir(pathdir_llarg)
        n = len(imgs)
        print(n)
        if nImgs == None:
            nImages = n
            
        this_patient = 0	
        while this_patient < min(nImages,n):
            print(len(imgs))
            if len(imgs) == 0:
                break
            index = random.randint(0,len(imgs)-1)
            imgname = imgs.pop(index)
            if imgname[-3:] != "png":
                continue
            
            imgpath = os.path.join(pathdir_llarg,imgname)
            
           
            
            img = cv2.imread(imgpath)
            print("imatge carregada")
            if img.shape != (256, 256, 3):
                imgs.pop(index)
                if len(imgs) == 0:
                    break
                print("Image with wrong size found", img.size)
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[-1] >= 3 else img[:, :, :3]
            diagnosis = excel.loc[excel['CODI'] == pathDir[:-2], 'DENSITAT'].values[0]

            if diagnosis == "NEGATIVA":
                diagnosis = -1
            elif diagnosis == "BAIXA" or diagnosis == "ALTA":
                diagnosis = 1
            
            patientsMeta.append([pathDir[:-2],imgname,diagnosis])
	 		# transpose img
            img = np.transpose(img, (2, 0, 1))
			#img = img.astype(np.float32)
            img = img.astype(np.float32)
            img = img/255.0
            patientsImgs.append(img)
            this_patient += 1

    return patientsImgs, patientsMeta