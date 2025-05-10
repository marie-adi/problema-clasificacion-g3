import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.multiclass import OneVsRestClassifier

df = pd.read_csv('data/df_final.csv')

# Crear un mapeo simple donde la clave y el valor son ambos el nombre del país
try:
    # Obtener todos los países únicos
    unique_entities = df['entity'].unique()
    
    # Crear un diccionario donde cada país se mapea a sí mismo
    entity_names = {country: country for country in unique_entities}
    
    print(f"Se han mapeado {len(entity_names)} países únicos.")
except Exception as e:
    print(f"Error al crear el mapeo de países: {e}")
    # Usar etiquetas genéricas como fallback
    unique_entities = df['entity'].unique()
    entity_names = {country: f"País {i+1}" for i, country in enumerate(unique_entities)}

# Análisis de depresión por país
print("\n--- Análisis de depresión por país ---")

# Agrupar por país y calcular estadísticas de depresión
country_depression = df.groupby('entity')['dalys_depressive_disorders'].agg(['mean', 'min', 'max', 'std']).reset_index()

# Añadir nombres de países
country_depression['country_name'] = country_depression['entity'].map(entity_names)

# Calcular percentiles para clasificar países
country_depression['percentile'] = country_depression['mean'].rank(pct=True) * 100

# Clasificar países en categorías según percentiles
country_depression['category'] = pd.qcut(
    country_depression['percentile'], 
    q=3, 
    labels=['Bajo', 'Medio', 'Alto']
)

# Ordenar por nivel medio de depresión (de mayor a menor)
country_depression = country_depression.sort_values('mean', ascending=False)

# Mostrar los resultados con la columna category incluida
print("\nNiveles de depresión por país (ordenados de mayor a menor):")
print(country_depression[['country_name', 'mean', 'min', 'max', 'std', 'category']])

# Visualizar los países con mayor y menor depresión
plt.figure(figsize=(12, 8))
top_countries = country_depression.head(10)  # Top 10 países con mayor depresión
sns.barplot(x='mean', y='country_name', hue='country_name', data=top_countries, palette='Reds_r', legend=False)
plt.title('Top 10 países con mayor nivel de depresión')
plt.xlabel('Nivel medio de DALYS por trastornos depresivos')
plt.ylabel('País')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
bottom_countries = country_depression.tail(10)  # 10 países con menor depresión
bottom_countries = bottom_countries.sort_values('mean')  # Ordenar ascendentemente para visualización
sns.barplot(x='mean', y='country_name', hue='country_name', data=bottom_countries, palette='Blues_r', legend=False)
plt.title('10 países con menor nivel de depresión')
plt.xlabel('Nivel medio de DALYS por trastornos depresivos')
plt.ylabel('País')
plt.tight_layout()
plt.show()

# Distribución de categorías de depresión
plt.figure(figsize=(10, 6))
category_counts = country_depression['category'].value_counts()
# Convertir a DataFrame para usar hue
category_df = pd.DataFrame({'Categoría': category_counts.index, 'Valor': category_counts.values})
sns.barplot(x='Categoría', y='Valor', hue='Categoría', data=category_df, palette='viridis', legend=False)
plt.title('Distribución de países por categoría de depresión')
plt.xlabel('Categoría de depresión')
plt.ylabel('Número de países')
plt.tight_layout()
plt.show()

# entrenar el modelo KNN
X = df.drop(['year', 'anxiety_disorders', 'eating_disorders',
             'dalys_eating_disorders', 'dalys_anxiety_disorders', 'entity'], axis=1)

# Añadir la categoría de depresión al dataframe original
df['country_name'] = df['entity'].map(entity_names)
df = df.merge(country_depression[['entity', 'category']], on='entity')

# Nuestra variable objetivo será la categoría de depresión del país
y = df['category']

# Verificar las dimensiones
print(f"\nDimensiones de X: {X.shape}")
print(f"Dimensiones de y: {y.shape}")

# Mostrar la distribución de las clases
print("\nDistribución de clases:")
print(y.value_counts())

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nX_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear y entrenar el modelo KNN para clasificación
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Hacer predicciones
y_pred = knn.predict(X_test_scaled)

# Evaluar el rendimiento del modelo
print("\nPrecisión del modelo:", accuracy_score(y_test, y_pred))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nInforme de clasificación:")
print(classification_report(y_test, y_pred))

# Optimizar el número de vecinos
error_rate = []

for i in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_scaled, y_train)
    pred_i = knn.predict(X_test_scaled)
    error_rate.append(1 - accuracy_score(y_test, pred_i))

# Visualizar el error para diferentes valores de k
plt.figure(figsize=(10, 6))
plt.plot(range(1, 31), error_rate, color='blue', linestyle='dashed',
         marker='o', markerfacecolor='red', markersize=10)
plt.title('Tasa de Error vs. Valor de K')
plt.xlabel('K')
plt.ylabel('Tasa de Error')
plt.tight_layout()
plt.show()

# Encontrar el valor de k con menor error
optimal_k = error_rate.index(min(error_rate)) + 1
print(f"\nEl valor óptimo de k es: {optimal_k}")

# Entrenar el modelo final con el k óptimo
final_knn = KNeighborsClassifier(n_neighbors=optimal_k)
final_knn.fit(X_train_scaled, y_train)
final_pred = final_knn.predict(X_test_scaled)
print(f"Precisión con k={optimal_k}: {accuracy_score(y_test, final_pred)}")

# Marcar que ya se encontró el k óptimo
optimal_k_found = True

# Analizar la importancia de las características
from sklearn.inspection import permutation_importance

result = permutation_importance(final_knn, X_test_scaled, y_test, n_repeats=10, random_state=42)
importance = result.importances_mean

# Mostrar la importancia de cada característica
feature_names = X.columns
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

print("\nImportancia de las características:")
print(feature_importance)

# Visualizar la importancia de las características
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importancia')
plt.ylabel('Característica')
plt.title('Importancia de las características en la predicción')
plt.tight_layout()
plt.show()

# Análisis adicional: Mapa de calor de correlación entre variables
plt.figure(figsize=(12, 10))
correlation = df[X.columns].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlación entre variables predictoras')
plt.tight_layout()
plt.show()

# Resumen final: Top 5 países con mayor y menor depresión
print("\n--- RESUMEN FINAL ---")
print("\nTop 5 países con MAYOR nivel de depresión:")
print(country_depression[['country_name', 'mean', 'category']].head(5))

print("\nTop 5 países con MENOR nivel de depresión:")
print(country_depression[['country_name', 'mean', 'category']].tail(5))

# 1. Evaluación básica del modelo
print("\n--- EVALUACIÓN DEL MODELO ---")

# Precisión en conjunto de entrenamiento vs conjunto de prueba
train_accuracy = accuracy_score(y_train, final_knn.predict(X_train_scaled))
test_accuracy = accuracy_score(y_test, final_pred)

print(f"Precisión en conjunto de entrenamiento: {train_accuracy:.4f}")
print(f"Precisión en conjunto de prueba: {test_accuracy:.4f}")
print(f"Diferencia (indicador de overfitting): {train_accuracy - test_accuracy:.4f}")

# Interpretación del overfitting
if train_accuracy - test_accuracy > 0.1:
    print("ALERTA: Posible overfitting. La precisión en entrenamiento es significativamente mayor que en prueba.")
elif train_accuracy - test_accuracy > 0.05:
    print("PRECAUCIÓN: Ligero overfitting. Hay una diferencia moderada entre entrenamiento y prueba.")
else:
    print("BUENO: No hay evidencia clara de overfitting. El modelo generaliza bien.")

# 2. Validación cruzada para una evaluación más robusta
cv_scores = cross_val_score(final_knn, X, y, cv=5)
print(f"\nPrecisión con validación cruzada (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Interpretación de la validación cruzada
if cv_scores.mean() < 0.6:
    print("ALERTA: El rendimiento del modelo es bajo según la validación cruzada.")
elif cv_scores.mean() < 0.75:
    print("ACEPTABLE: El rendimiento del modelo es moderado según la validación cruzada.")
else:
    print("BUENO: El rendimiento del modelo es alto según la validación cruzada.")

# 3. Curvas de aprendizaje para detectar overfitting/underfitting
train_sizes, train_scores, test_scores = learning_curve(
    final_knn, X, y, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

# Calcular medias y desviaciones estándar
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Graficar curvas de aprendizaje
plt.figure(figsize=(10, 6))
plt.grid()
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Precisión en entrenamiento")
plt.plot(train_sizes, test_mean, 'o-', color="g", label="Precisión en validación cruzada")
plt.xlabel("Tamaño del conjunto de entrenamiento")
plt.ylabel("Precisión")
plt.title("Curvas de aprendizaje para KNN")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# Interpretación de las curvas de aprendizaje
gap = train_mean[-1] - test_mean[-1]
if gap > 0.1:
    print("\nCURVAS DE APRENDIZAJE: Indican overfitting. Hay una brecha significativa entre entrenamiento y validación.")
    print("RECOMENDACIÓN: Considera usar menos vecinos, aplicar regularización o recopilar más datos.")
elif test_mean[-1] < 0.7:
    print("\nCURVAS DE APRENDIZAJE: Indican underfitting. El modelo no está capturando bien los patrones.")
    print("RECOMENDACIÓN: Considera usar más vecinos, incluir más características o usar un modelo más complejo.")
else:
    print("\nCURVAS DE APRENDIZAJE: El modelo parece estar bien equilibrado sin overfitting ni underfitting significativos.")

# 4. Análisis ROC y Precision-Recall para evaluación multiclase
# Binarizar las etiquetas para análisis ROC multiclase
y_test_bin = label_binarize(y_test, classes=np.unique(y))
n_classes = y_test_bin.shape[1]

# Configurar clasificador One-vs-Rest
classifier = OneVsRestClassifier(final_knn)

# Obtener probabilidades de predicción, KNN no tiene predict_proba por defecto, usamos distance para aproximar
y_score = classifier.fit(X_train_scaled, y_train).predict_proba(X_test_scaled)

# Calcular curva ROC y AUC para cada clase
fpr = dict()
tpr = dict()
roc_auc = dict()

plt.figure(figsize=(10, 8))
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.plot(fpr[i], tpr[i], lw=2,
             label=f'ROC clase {list(np.unique(y))[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC para clasificación multiclase')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# Calcular curvas Precision-Recall para cada clase
precision = dict()
recall = dict()
avg_precision = dict()

plt.figure(figsize=(10, 8))
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    avg_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])
    
    plt.plot(recall[i], precision[i], lw=2,
             label=f'P-R clase {list(np.unique(y))[i]} (AP = {avg_precision[i]:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curvas Precision-Recall para clasificación multiclase')
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# 5. Resumen final de la evaluación
print("\n--- RESUMEN DE EVALUACIÓN DEL MODELO ---")
print(f"Precisión general: {test_accuracy:.4f}")
print(f"Precisión con validación cruzada: {cv_scores.mean():.4f}")
print(f"Diferencia entre entrenamiento y prueba: {gap:.4f}")

# Calcular AUC promedio
mean_auc = np.mean(list(roc_auc.values()))
print(f"AUC promedio: {mean_auc:.4f}")

# Evaluación final
if test_accuracy > 0.8 and gap < 0.1 and mean_auc > 0.8:
    print("\nEVALUACIÓN FINAL: EXCELENTE. El modelo tiene alta precisión, generaliza bien y discrimina bien entre clases.")
elif test_accuracy > 0.7 and gap < 0.15 and mean_auc > 0.7:
    print("\nEVALUACIÓN FINAL: BUENO. El modelo tiene precisión aceptable con ligero overfitting.")
elif test_accuracy > 0.6 and gap < 0.2:
    print("\nEVALUACIÓN FINAL: ACEPTABLE. El modelo funciona mejor que el azar pero tiene margen de mejora.")
else:
    print("\nEVALUACIÓN FINAL: DEFICIENTE. El modelo necesita mejoras significativas o reconsiderar el enfoque.")

# 6. Sugerencias de mejora
print("\n--- SUGERENCIAS DE MEJORA ---")
if gap > 0.1:
    print("- Reducir overfitting: Disminuir el número de vecinos o aplicar técnicas de regularización.")
if test_accuracy < 0.7:
    print("- Mejorar precisión general: Considerar características adicionales o transformaciones de datos.")
if cv_scores.std() > 0.1:
    print("- Reducir variabilidad: El modelo es sensible a la partición de datos. Considerar más datos o técnicas de ensamblaje.")
if mean_auc < 0.7:
    print("- Mejorar discriminación: El modelo tiene dificultades para distinguir entre clases. Considerar balancear clases o usar pesos.")

print("\nRecuerda que estos umbrales son orientativos y deben ajustarse según el contexto específico del problema.")

# Marcar que ya se realizó la evaluación
model_evaluation_done = True