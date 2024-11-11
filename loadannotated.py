import glob
import os
import random
import cv2
import pandas as pd
import numpy as np

#tenim un excel HP_WSI-CoordAllAnnotatedPatches.xlsx" amb:
#Pat_ID: B22-129
#Section_ID: 0
#ON PAT_ID_SECTION_ID ÉS EL NOM DE LA CARPETA AMB LA IMATGE AMB .PNG
#ON EL FILENAME ËS WINDOW_ID. 

def loadAnnotated(ListFolders, nImages):
    pathExcel = "HP_WSI-CoordAllAnnotatedPatches.xlsx"
    excel = pd.read_excel(pathExcel)
    
    annotatedImgs = np.array(0)
    
    for pathDir in ListFolders:
        # extreure Pat_ID i Section_ID del nom de la carpeta
        folder_name = os.path.basename(pathDir)
        pat_id, section_id = folder_name.split('_')

        # Filtrar del excel les files amb aquest Pat_ID i Section_ID
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
            img_window_id = os.path.basename(img_path).replace('.png', '')

            #fer match window_id
            presence_row = folder_data[folder_data['Window_ID'] == int(img_window_id)]
            #extreure valor presence
            presence = presence_row['Presence'].values[0]

            # Crear metadata
            metadates = [pat_id, img_window_id, presence]
            annotatedImgs.append([img, metadates])

    return annotatedImgs

