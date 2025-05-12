import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import numpy as np

# Cargar los datos
df = pd.read_csv('Data/df_final_encoded.csv')

# Eliminar filas donde la variable objetivo tiene valores NaN
df = df.dropna(subset=['dalys_depressive_disorders'])

# Separar features (X) y variable objetivo (y)
X = df.drop(['dalys_depressive_disorders'], axis=1)
y = df['dalys_depressive_disorders']

# Crear un imputador para las características
imputer = SimpleImputer(strategy='mean')

# Aplicar el imputador a X
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo SVM
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)

# Realizar predicciones
y_pred = svm_model.predict(X_test)

# Evaluar el modelo
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# Calcular la precisión del modelo
accuracy = svm_model.score(X_test, y_test)
print(f"\nPrecisión del modelo: {accuracy:.2f}")