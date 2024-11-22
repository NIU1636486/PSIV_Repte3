import numpy as np
import cv2
from sklearn.metrics import roc_curve


import matplotlib.pyplot as plt

# Entrenar autoencoder i fer-ho servir per reconstruct imatges del annotated dataset
#per cada annotated image, calcular la pixel-wise reconstruction error = Fred i guardar valor amb el seu true label pel ROC

#Els píxels vermellosos es calculen aplicant un filtre en l’espai de color HSV. 
# En aquest espai de color, els píxels amb presència de H. pylori tenen un matís en el rang [−20, 20],
# per la qual cosa l’àrea de píxels vermellosos es defineix com el nombre de píxels amb un matís en [−20, 20].

#calcular Fred com el ratio de pixels en original i reconstructed
#Representa la mesura de anomalous staining en el red channel
def calculate_f_red(original_img, reconstructed_img, threshold_low=-20, threshold_high=20):
    #convertir imatges a HSV i extreure HUE channel
    original_hue = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)[:, :, 0]
    reconstructed_hue = cv2.cvtColor(reconstructed_img, cv2.COLOR_BGR2HSV)[:, :, 0]
    
    # Convert hue range to 0-179 scale used in OpenCV
    # perque en OpenCV  el hue es represemtat en escala 0-179. Per map negative map values (-20) a valid range
    hue_low = (threshold_low + 180) % 180
    hue_high = (threshold_high + 180) % 180
    
    # comptar pixels vermells, tractant el rang circular del Hue range
    if hue_low < hue_high:
        original_red_pixels = np.sum((original_hue >= hue_low) & (original_hue <= hue_high))
        reconstructed_red_pixels = np.sum((reconstructed_hue >= hue_low) & (reconstructed_hue <= hue_high))
    else:
        original_red_pixels = np.sum((original_hue >= hue_low) | (original_hue <= hue_high))
        reconstructed_red_pixels = np.sum((reconstructed_hue >= hue_low) | (reconstructed_hue <= hue_high))
    
    ## Avoid division by zero in case no red pixels exist in the original image
    #return a default high value
    #indicating no loss since there was no red to lose.
    if original_red_pixels == 0:
        return 0.0
        
    # Calculate the fraction of reddish pixels preserved
    f_red = reconstructed_red_pixels / original_red_pixels
    return f_red

#funcio roc_threshold_analysis amb collected errors(fred) i labels per determinar optimal threshold (+ proper a 0,1)
#apply aquest threshol durant test/evaluation per classificar patches
def roc_threshold_analysis(errors, labels):
    binary_labels = np.array([1 if lbl == 1 else 0 for lbl in labels])  # Map -1 to 0 and 1 to 1
    #Ambiguous Labels (0): Excluded from the analysis to avoid contaminating the ROC curve.
    #roc curve
    fpr, tpr, thresholds = roc_curve(binary_labels, errors)
    
    #distancies al punt (0,1)
    distances = np.sqrt((fpr - 0)**2 + (1 - tpr)**2)
    
    #el threshold + optim = el + proper 
    optimal_idx = np.argmin(distances)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold, fpr, tpr

