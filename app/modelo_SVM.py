import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Cargar los datos
df = pd.read_csv('../Data/df_final.csv')

# Filtrar solo los datos del 2019
df = df[df['year'] == 2019].copy()

# Separar features (X) y variable objetivo (y)
X = df.drop(['year', 'anxiety_disorders', 'eating_disorders', 
             'dalys_eating_disorders', 'dalys_anxiety_disorders', 'entity'], axis=1)

# Calcular el percentil 75 para una clasificación más estricta
umbral = df['depressive_disorders'].quantile(0.60)

# Convertir la variable objetivo a binaria (0 o 1) usando el percentil 75
y = (df['depressive_disorders'] > umbral).astype(int)

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Definir los parámetros para la búsqueda en cuadrícula
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['rbf', 'linear']
}

# Crear el modelo base
base_model = SVC(class_weight='balanced', random_state=42)

# Crear y ejecutar la búsqueda en cuadrícula
grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Entrenar el modelo con búsqueda en cuadrícula
print("Iniciando búsqueda de mejores parámetros...")
grid_search.fit(X_train, y_train)

# Mostrar los mejores parámetros encontrados
print("\nMejores parámetros encontrados:")
print(grid_search.best_params_)

# Obtener el mejor modelo
best_model = grid_search.best_estimator_

# Realizar predicciones con el mejor modelo
y_pred = best_model.predict(X_test)

# Evaluar el modelo mejorado
print("\nResultados con el modelo mejorado:")
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

precision = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo mejorado: {precision:.4f}")

# Evaluar el modelo mejorado
print("\nResultados con el modelo mejorado:")
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['Bajo riesgo', 'Alto riesgo']))

precision = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo: {precision:.2f}")

# Crear un diccionario de resultados
resultados_binarios = {}
for idx, row in df.iterrows():
    valor_depresion = row['depressive_disorders']
    prediccion = y[idx]
    resultados_binarios[row['entity']] = {
        'riesgo_binario': int(prediccion),
        'categoria': 'Alto riesgo' if prediccion == 1 else 'Bajo riesgo',
        'valor_depresion': float(valor_depresion),
        'año': 2019
    }

# Análisis de resultados
total_paises = len(resultados_binarios)
paises_alto_riesgo = sum(1 for pais in resultados_binarios.values() if pais['riesgo_binario'] == 1)
paises_bajo_riesgo = sum(1 for pais in resultados_binarios.values() if pais['riesgo_binario'] == 0)

print("\nAnálisis de países:")
print(f"Total de países procesados: {total_paises}")
print(f"Países con alto riesgo: {paises_alto_riesgo}")
print(f"Países con bajo riesgo: {paises_bajo_riesgo}")

# Guardar resultados binarios en JSON
import json
import os

os.makedirs('resultados', exist_ok=True)
output_file = 'resultados/predicciones_binarias.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(resultados_binarios, f, indent=4, ensure_ascii=False)

print(f"\nPredicciones binarias guardadas en: {output_file}")

# Calcular el overfitting
train_score = best_model.score(X_train, y_train)
test_score = best_model.score(X_test, y_test)
overfitting = train_score - test_score

print("\nAnálisis de Overfitting:")
print(f"Score en entrenamiento: {train_score:.4f}")
print(f"Score en prueba: {test_score:.4f}")
print(f"Diferencia (overfitting): {overfitting:.4f}")

# Si el overfitting es mayor a 0.1, podría indicar un problema
if overfitting > 0.1:
    print("¡Advertencia! El modelo muestra señales de overfitting significativo")
elif overfitting > 0.05:
    print("El modelo muestra señales moderadas de overfitting")
else:
    print("El modelo no muestra señales significativas de overfitting")

# Guardar el modelo
modelo_file = 'modelo_svm.pkl'
with open(modelo_file, 'wb') as file:
    pickle.dump(best_model, file)

print(f"\nModelo guardado en: {modelo_file}")

# Crear dos subplots
plt.figure(figsize=(15, 5))

# Después de obtener la precisión del modelo mejorado, añadimos la curva de aprendizaje
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=5,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10)):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Tamaño del conjunto de entrenamiento")
    plt.ylabel("Puntuación")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring='accuracy')
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Puntuación de entrenamiento")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Puntuación de validación cruzada")

    plt.legend(loc="best")
    return plt

# Crear dos subplots
plt.figure(figsize=(15, 5))

# Primer subplot: Resultados de la búsqueda en cuadrícula
plt.subplot(1, 2, 1)
results = pd.DataFrame(grid_search.cv_results_)
plt.scatter(results['param_C'], results['mean_test_score'], c='blue', alpha=0.5)
plt.xscale('log')
plt.xlabel('Valor de C')
plt.ylabel('Precisión media en validación cruzada')
plt.title('Resultados de la búsqueda en cuadrícula')
plt.grid(True)

# Segundo subplot: Curva de aprendizaje
plt.subplot(1, 2, 2)
plot_learning_curve(
    best_model, 
    "Curva de Aprendizaje",
    X_scaled, 
    y, 
    ylim=(0.0, 1.1),
    cv=5,
    n_jobs=-1
)

plt.tight_layout()
plt.show()