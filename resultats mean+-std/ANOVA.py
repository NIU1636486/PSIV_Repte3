import pandas as pd
import numpy as np
from scipy.stats import f_oneway

# dades: mean i std de cada config 
data = {
    'Config': ['Config1'] * 3 + ['Config2'] * 3 + ['Config3'] * 3,
    'Metric': ['Precision', 'Recall', 'F1-score'] * 3,
    'Negative_Pylori': [0.75, 0.74, 0.71, 0.79, 0.78, 0.76, 0.78, 0.78, 0.75],
    'Positive_Pylori': [0.72, 0.74, 0.68, 0.77, 0.81, 0.74, 0.76, 0.79, 0.72]
}

df = pd.DataFrame(data)

#Anova per cada classe i mètrica
for metric in ['Negative_Pylori', 'Positive_Pylori']:
    print(f"\n### ANOVA para {metric} ###")
    config1 = df[df['Config'] == 'Config1'][metric]
    config2 = df[df['Config'] == 'Config2'][metric]
    config3 = df[df['Config'] == 'Config3'][metric]

    #Anova de una via: ja que 3 metodes
    #Hipitesis nula: mitjana son iguals
    #Hipotesis alternativa: al menys una és diferent
    #fstat = variabilidad entre grups/variabilitat dins grups. Si és alt esq variab entre grups > variab dins = medias dif
    f_stat, p_value = f_oneway(config1, config2, config3)
    print(f"F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")

    # Si p_valor <= rebutjem Ho
    #si p_valor >= no rebutjem HO que diu mitjanes iguals 
    if p_value < 0.05:
        print("Hi ha dif significatives")
    else:
        print("No hi ha dif significatives")

### ANOVA para Negative_Pylori ###
#F-statistic: 5.0690, p-value: 0.0514
#No hi ha dif significatives

### ANOVA para Positive_Pylori ###
#F-statistic: 2.5392, p-value: 0.1589
#No hi ha dif significatives