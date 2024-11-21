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

def loadAnnotated(ListFolders, nImages):
    pathExcel = "/Users/carlotacortes/Desktop/HP_WSI-CoordAllAnnotatedPatches.xlsx"
    excel = pd.read_excel(pathExcel)
    # AGAFAR WINDOW ID EXCEL
    excel['Window_ID'] = excel['Window_ID'].astype(str).str.strip()
    
    annotatedImgs = []
    annotatedtMeta= []
    
    for pathDir in ListFolders:
        # extreure Pat_ID i Section_ID del nom de la carpeta
        folder_name = os.path.basename(pathDir)
        pat_id, section_id = folder_name.split('_')
        
        # filtrar del excel els que coincideixen amb pat id i section id carpeta
        folder_data = excel[(excel['Pat_ID'] == pat_id) & (excel['Section_ID'] == int(section_id))]
        
        # de la carpeta aconseguir totes les imatges
        imgs = glob.glob(os.path.join(pathDir, '*.png'))
        n = len(imgs)

        for _ in range(nImages):
            # seleccionar una imatge de manera random
            index = random.randint(0, n-1)
            img_path = imgs[index]
            img = cv2.imread(img_path)

            # extreure filename sense el .png per fer match amb Window_ID del excel
            # Extract filename without extension using os.path.splitext
            img_filename = os.path.splitext(os.path.basename(img_path))[0]

            # Match filename with Window_ID in Excel (Remove leading zeros)
            img_filename_cleaned = re.sub(r'^0+', '', img_filename.strip())
            
            
            # fer match Window_ID amb folder data
            presence_row = folder_data[folder_data['Window_ID'] == img_filename_cleaned]
            if presence_row.empty:
                print(f"No matching Window_ID for image '{img_filename_cleaned}' in Excel data. Skipping.")
                continue
            
            #extreure valor presence
            presence = presence_row['Presence'].values[0]

            #afegir a les dues llistes
            annotatedImgs.append(img)
            annotatedtMeta.append([pat_id, img_filename_cleaned, presence])

    return annotatedImgs, annotatedtMeta

ListFolders = glob.glob("/Users/carlotacortes/Desktop/Annotated/*")

patientsImgs, patientsMeta  = loadAnnotated(ListFolders, nImages=2)

# Print the results for verification
for img, metadata in zip(patientsImgs, patientsMeta):
    #print(img)
    print("Metadata:", metadata)
