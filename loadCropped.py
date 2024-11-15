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
	pathExcel = "/fhome/maed/HelicoDataSet/PatientDiagnosis.csv"
	patientsImgs = list()
	patientsMeta= []
	excel = pd.read_csv(pathExcel)
	print(excel)
	for pathDir in ListFolders:
		pathdir_llarg = os.path.join("/fhome/maed/HelicoDataSet/Cropped/", pathDir)
		imgs = os.path.join(pathdir_llarg,'*.png')
		imgs = glob.glob(imgs)
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
	 		# transpose img
			img = np.transpose(img, (2, 0, 1))
			patientsImgs.append(img)

	return patientsImgs, patientsMeta
    
    