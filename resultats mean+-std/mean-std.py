
import pandas as pd

file_path = '/Users/carlotacortes/Desktop/metriques_config1ex.xlsx'  # Reemplaza con la ruta real de tu archivo
df = pd.read_excel(file_path)

# Rename columns per accessibilitat
df.columns = [
    'Fold', 'Precision pos', 'Precision neg',
    'Recall pos', 'Recall neg', 'F1-score pos', 'F1-score neg'
]

# eliminar si hi ha alguna fila sense res
df = df.iloc[1:].reset_index(drop=True)

# canviar a numeriques
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# statistical summary 
summary = {
    "Statistic": ["Precision", "Recall", "F1-score"],
    "Positive H.pylori": [
        f"{df['Precision pos'].mean():.2f} ± {df['Precision pos'].std():.2f}",
        f"{df['Recall pos'].mean():.2f} ± {df['Recall pos'].std():.2f}",
        f"{df['F1-score pos'].mean():.2f} ± {df['F1-score pos'].std():.2f}",
    ],
    "Negative H.pylori": [
        f"{df['Precision neg'].mean():.2f} ± {df['Precision neg'].std():.2f}",
        f"{df['Recall neg'].mean():.2f} ± {df['Recall neg'].std():.2f}",
        f"{df['F1-score neg'].mean():.2f} ± {df['F1-score neg'].std():.2f}",
    ],
}

summary_df = pd.DataFrame(summary)
print(summary_df.head())