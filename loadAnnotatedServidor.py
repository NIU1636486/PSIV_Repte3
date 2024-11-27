# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 19:06:07 2024

@author: marcp
"""

import glob
import os
import random
import cv2
import pandas as pd
import numpy as np
import re

#tenim un excel HP_WSI-CoordAllAnnotatedPatches.xlsx" amb:
#Pat_ID: B22-129
#Section_ID: 0
#ON PAT_ID_SECTION_ID ÉS EL NOM DE LA CARPETA AMB LA IMATGE AMB .PNG
#ON EL FILENAME ËS WINDOW_ID. 

def loadAnnotated(ListFolders, nImg):
    
    pathExcel = "/fhome/maed/HelicoDataSet/HP_WSI-CoordAllAnnotatedPatches.xlsx"
    excel = pd.read_excel(pathExcel)
    # ELIMINAR SI WINDOW ID CONTE number_Aug???
    excel = excel[~excel['Window_ID'].astype(str).str.contains('_Aug', regex=False)]
    # nomes numerics i passar-los a integer per no llegir-los com a float
    excel = excel[excel['Window_ID'].apply(lambda x: str(x).isdigit())]
    excel['Window_ID'] = excel['Window_ID'].astype(int)
    
    annotatedImgs = []
    annotatedtMeta= []
    nImages = nImg
    
    for pathDir in ListFolders:
        # extreure Pat_ID i Section_ID del nom de la carpeta
        folder_name = os.path.basename(pathDir)
        pat_id, section_id = folder_name.split('_')
        
        # filtrar del excel els que coincideixen amb pat id i section id carpeta
        folder_data = excel[(excel['Pat_ID'] == pat_id) & (excel['Section_ID'] == int(section_id))]
        
        pathdir_llarg = os.path.join("/fhome/maed/HelicoDataSet/CrossValidation/Annotated/", pathDir)
        
        
        pathimgs = os.path.join(pathdir_llarg,'*.png')
        imgs = glob.glob(pathimgs)
        n = len(imgs)
        
        if nImg is None:
            nImages = n

        this_patient = 0
        while this_patient < min(nImages, n):
            print(len(imgs))
            if len(imgs) == 0:
                break
            
            index = random.randint(0,len(imgs)-1)
            imgname = imgs.pop(index)
            print(imgname)
            img = cv2.imread(imgname)

            # extreure filename sense el .png per fer match amb Window_ID del excel
            img_filename = os.path.basename(imgname).replace('.png', '')
            
            # Use regex to extract only the numeric part of the filename (Window_ID)
            match = re.match(r"(\d+)", img_filename)

            if not match:
                print(f"Filename '{img_filename}.png' does not match expected pattern. Skipping.")
                continue
            
            # Convert the numeric part to an integer
            img_window_id = int(match.group(1))
            
            
            # fer match Window_ID amb folder data
            presence_row = folder_data[folder_data['Window_ID'] == img_window_id]
            if presence_row.empty:
                print(f"No matching Window_ID for image '{img_filename}.png' in Excel data. Skipping.")
                presence = -1
            else:
                #extreure valor presence
                presence = presence_row['Presence'].values[0]
            if presence != 0:
                #afegir a les dues llistes
                annotatedImgs.append(img)
                annotatedtMeta.append([pat_id, img_window_id, presence])
                this_patient +=1
    print("Dades anotated dins")
    return annotatedImgs, annotatedtMeta

#ListFolders = glob.glob("/Users/carlotacortes/Desktop/Annotated/*")

#patientsImgs, patientsMeta  = loadAnnotated(ListFolders, nImages)

# Print the results for verification
#for img, metadata in zip(patientsImgs, patientsMeta):
    #print(img)
    #print("Metadata:", metadata)