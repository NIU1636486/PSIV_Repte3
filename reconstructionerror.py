import numpy as np
import cv2
from sklearn.metrics import roc_curve

# Entrenar autoencoder i fer-ho servir per reconstruct imatges del annotated dataset
#per cada annotated image, calcular el reconstruction error amb funcio: compute_reconstruction_error

#per cada imatge calcular Fred i guardar valor amb el seu true label pel ROC

#funcio roc_threshold_analysis amb collected errors i labels per determinar optimal threshold

#apply aquest threshol durant test/evaluation per classificar patches

#compute the pixel-wise reconstruction error
#centrarse en el red channel del HSV
#calcula la diferencia absoluta
def compute_reconstruction_error(original_img, reconstructed_img):
    # convertir imatges a espai HSV 
    original_hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
    reconstructed_hsv = cv2.cvtColor(reconstructed_img, cv2.COLOR_BGR2HSV)
    
    #extreure el red channel (H)
    original_hue = original_hsv[:, :, 0]
    reconstructed_hue = reconstructed_hsv[:, :, 0]
    
    # calcular dif. absoluta entre original i reconstructed
    error = np.abs(original_hue - reconstructed_hue)
    return error


#calcular Fred com el ratio de pixels dins del threshold specific en original i reconstructed
#Representa la mesura de anomalous staining en el red channel
def calculate_f_red(original_img, reconstructed_img, threshold_low=-20, threshold_high=20):
    original_hue = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)[:, :, 0]
    reconstructed_hue = cv2.cvtColor(reconstructed_img, cv2.COLOR_BGR2HSV)[:, :, 0]
    
    # comptar pixels dins del interval de threshold 
    original_count = np.sum((original_hue > threshold_low) & (original_hue < threshold_high))
    reconstructed_count = np.sum((reconstructed_hue > threshold_low) & (reconstructed_hue < threshold_high))
    
    # evitar divisio entre 0
    if reconstructed_count == 0:
        return 0
    return original_count / reconstructed_count

#pren una llista de reconstruction errors i les seves etiquetes corresponents
#fa la ROC curve i selecciona el threshold optim- el més proper a (0,1) 
def roc_threshold_analysis(errors, labels):
    #roc curve
    fpr, tpr, thresholds = roc_curve(labels, errors)
    
    #distancies al punt (0,1)
    distances = np.sqrt((fpr - 0)**2 + (1 - tpr)**2)
    
    #el threshold + optim = el + proper 
    optimal_idx = np.argmin(distances)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold, fpr, tpr
