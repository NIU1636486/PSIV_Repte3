import pickle
import matplotlib.pyplot as plt
# Importem llibreries per la validació
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import roc_curve
# Ruta del arxiu pickle
archivo_pickle = '/users/carlotacortes/Desktop/validation_200imgsConfig2.pkl'

# Abrir y cargar el archivo
with open(archivo_pickle, 'rb') as file:
    datos = pickle.load(file)

print("\nANALISI ROC TRAINING THRESHOLD PATCH CLASSIFICATION")
OTPC = datos["OTPC"]
OTPD = datos["OTPD"]
RTPC = datos["RTPC"]
RTPD = datos["RTPD"]
for i,fold in enumerate(OTPC):
    OT, fpr, tpr = fold
    
    # Mostrem la corba ROC de l'analisi de l'entrenament del fold actual
    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve (Patch Classification")
    plt.scatter([0], [1], color="red", label="Ideal Point")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f'ROC Curve (Patch Classification) Fold {i+1}')
    plt.legend()
    plt.show()
    
print("\nANALISI ROC TRAINING THRESHOLD PATIENT DIAGNOSIS")

for i,fold in enumerate(OTPD):
    OT, fpr, tpr = fold
    
    # Mostrem la corba ROC de l'analisi de l'entrenament del fold actual
    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve (Patient Diagnosis")
    plt.scatter([0], [1], color="red", label="Ideal Point")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f'ROC Curve (Patient Diagnosis) Fold {i+1}')
    plt.legend()
    plt.show()
    
for i, fold in enumerate(RTPC):
    print("\nFOLD")
    # Dividim la classificació en reals i predites
    realLabels = [item[0] for item in fold]
    predLabels = [item[1] for item in fold]

    # Calculem les metriques de validació

    precision = precision_score(realLabels, predLabels, pos_label= 1)  
    recall = recall_score(realLabels, predLabels, pos_label= 1)
    f1 = f1_score(realLabels, predLabels, pos_label = 1)
    confusionMatrix = confusion_matrix(realLabels, predLabels)
    
    # Guardem les metriques de validació
    
    #PrecisionPatientsDiagnosis.append(precision)
    #RecallPatientsDiagnosis.append(recall)
    #F1ScorePatientsDiagnosis.append(f1)
    #ConfusionMatrixPatientsDiagnosis.append(confusionMatrix)
    
    # Imprimir resultados
    print(f"Precisió (Precision): {precision}")
    print(f"Sensibilitat (Recall): {recall}")
    print(f"F1-score: {f1}")
    print("Matriu de Confusió:")
    print(confusionMatrix)
    
    # Reporte detallado
    print("\nReporte de clasificación:")
    print(classification_report(realLabels, predLabels, labels=[1, -1]))
