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
    # ELIMINAR SI WINDOW ID CONTE number_Aug???
    excel = excel[~excel['Window_ID'].astype(str).str.contains('_Aug', regex=False)]
    
    # nomes numerics i passar-los a integer per no llegir-los com a float
    excel = excel[excel['Window_ID'].apply(lambda x: str(x).isdigit())]
    excel['Window_ID'] = excel['Window_ID'].astype(int)
    
    annotatedImgs = []
    
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
            img_filename = os.path.basename(img_path).replace('.png', '')
            
            # Use regex to extract only the numeric part of the filename (Window_ID)
            match = re.match(r"(\d+)", img_filename)

            # Convert the numeric part to an integer
            img_window_id = int(match.group(1))

            # Debugging 
            #print(f"Checking image '{img_filename}.png' with extracted Window_ID: {img_window_id}")
            #print("Available Window_IDs in folder data:", folder_data['Window_ID'].tolist())

            # fer match Window_ID amb folder data
            presence_row = folder_data[folder_data['Window_ID'] == img_window_id]
            if presence_row.empty:
                print(f"No matching Window_ID for image '{img_filename}.png' in Excel data. Skipping.")
                continue
            
            #extreure valor presence
            presence = presence_row['Presence'].values[0]

            # Crear metadata
            metadates = [pat_id, img_window_id, presence]
            annotatedImgs.append([img, metadates])

    return annotatedImgs

ListFolders = glob.glob("/Users/carlotacortes/Desktop/Annotated/*")

results = loadAnnotated(ListFolders, nImages=2)

# Print the results for verification
for img, metadata in results:
    print("Metadata:", metadata)
