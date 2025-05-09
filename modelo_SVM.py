import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos
df = pd.read_csv('Data/df_final_encoded.csv')

# Separar features (X) y variable objetivo (y)
X = df.drop(['year', 'anxiety_disorders', 'eating_disorders' ,'dalys_depressive_disorders', 'dalys_eating_disorders', 'dalys_anxiety_disorders', 'entity_encoded'], axis=1)
y = df['dalys_depressive_disorders']

# Convertir a booleanos usando la mediana como punto de corte
y_bool = y > y.median()

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Selección de características
selector = SelectKBest(f_classif, k=3)
X_selected = selector.fit_transform(X_scaled, y_bool)

# Obtener los nombres de las características seleccionadas
selected_features_mask = selector.get_support()
selected_features = X.columns[selected_features_mask].tolist()
print("\nCaracterísticas más importantes:")
print(selected_features)

# Crear el modelo SVM con parámetros optimizados
svm_model = SVC(
    C=0.1,
    kernel='rbf',
    gamma='scale',
    class_weight='balanced',
    random_state=42
)

# Configurar la validación cruzada
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Realizar validación cruzada
cv_scores = cross_val_score(svm_model, X_selected, y_bool, cv=kfold)

# Imprimir resultados de la validación cruzada
print("\nResultados de la validación cruzada:")
print(f"Precisión media: {cv_scores.mean():.4f}")
print(f"Desviación estándar: {cv_scores.std():.4f}")

# Dividir datos para evaluación final
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_bool, test_size=0.2, random_state=42)

# Entrenar el modelo final
svm_model.fit(X_train, y_train)

# Evaluar en conjunto de entrenamiento y prueba
train_pred = svm_model.predict(X_train)
test_pred = svm_model.predict(X_test)

train_accuracy = svm_model.score(X_train, y_train)
test_accuracy = svm_model.score(X_test, y_test)

print("\nAnálisis de Overfitting:")
print(f"Precisión en entrenamiento: {train_accuracy:.4f}")
print(f"Precisión en prueba: {test_accuracy:.4f}")
print(f"Diferencia (overfitting gap): {train_accuracy - test_accuracy:.4f}")

# Imprimir matriz de confusión con etiquetas booleanas
print("\nMatriz de Confusión:")
conf_matrix = confusion_matrix(y_test, test_pred)
print("                  Predicho False  Predicho True")
print(f"Real False    |      {conf_matrix[0][0]}           {conf_matrix[0][1]}")
print(f"Real True     |      {conf_matrix[1][0]}           {conf_matrix[1][1]}")

# Calcular y graficar la curva de aprendizaje
train_sizes, train_scores, test_scores = learning_curve(
    svm_model, X_selected, y_bool,
    train_sizes=np.linspace(0.3, 1.0, 10),
    cv=5, n_jobs=-1
)

# Calcular medias y desviaciones estándar
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Crear subplots para visualización
plt.figure(figsize=(12, 5))

# Subplot 1: Curva de aprendizaje
plt.subplot(1, 2, 1)
plt.plot(train_sizes, train_mean, label='Entrenamiento')
plt.plot(train_sizes, test_mean, label='Validación')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
plt.xlabel('Tamaño del conjunto de entrenamiento')
plt.ylabel('Puntuación')
plt.title('Curva de Aprendizaje')
plt.legend(loc='best')
plt.grid(True)

# Subplot 2: Gráfico de barras de precisión
plt.subplot(1, 2, 2)
plt.bar(['Entrenamiento', 'Prueba'], [train_accuracy, test_accuracy])
plt.ylabel('Precisión')
plt.title('Comparación de Precisión')
plt.grid(True)

plt.tight_layout()
plt.show()

# Análisis detallado del overfitting
overfitting_threshold = 0.1
overfitting_gap = train_accuracy - test_accuracy

print("\nAnálisis detallado del overfitting:")
print("-" * 50)
print(f"Gap de overfitting: {overfitting_gap:.4f}")

if overfitting_gap > overfitting_threshold:
    print("ADVERTENCIA: Se detecta overfitting significativo")
    print("\nPosibles soluciones:")
    print("1. Reducir la complejidad del modelo (disminuir C)")
    print("2. Aumentar la regularización")
    print("3. Reducir el número de características")
    print("4. Obtener más datos de entrenamiento")
else:
    print("El modelo no muestra signos significativos de overfitting")

print("\nMétricas de variabilidad:")
print(f"Desviación estándar en entrenamiento: {train_std.mean():.4f}")
print(f"Desviación estándar en validación: {test_std.mean():.4f}")

# ... existing code ...

print("\nMétricas de variabilidad:")
print(f"Desviación estándar en entrenamiento: {train_std.mean():.4f}")
print(f"Desviación estándar en validación: {test_std.mean():.4f}")

# Guardar el modelo entrenado
import joblib

# Guardar el modelo
joblib.dump(svm_model, 'model_svm.pkl')
print("\nModelo guardado como 'model_svm.pkl'")

# Guardar el scaler y el selector de características
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selector, 'selector.pkl')
print("Scaler y selector guardados como 'scaler.pkl' y 'selector.pkl'")