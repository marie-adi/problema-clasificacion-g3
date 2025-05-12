import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

def decodificar_prediccion(prediccion):
    # Diccionario de decodificación de predicciones
    decodificacion = {
        0: "No presenta niveles significativos de depresión",
        1: "Presenta niveles significativos de depresión"
    }
    print(f"Valor de predicción recibido: {prediccion}")
    return decodificacion.get(prediccion, f"Predicción desconocida (valor: {prediccion})")

def test_modelo_svm():
    # Cargar el modelo entrenado
    try:
        with open('../pkl/modelo_svm.pkl', 'rb') as archivo:
            modelo = pickle.load(archivo)
        print("Modelo cargado exitosamente")
        
        # Cargar datos del 2019
        df = pd.read_csv('../Data/df_final.csv')
        df_2019 = df[df['year'] == 2019].copy()
        
        print("\nPaíses disponibles:")
        for pais in df_2019['entity']:
            print(f"- {pais}")
        
        try:
            # Solicitar entrada del país
            pais = input("\nIngrese el nombre del país: ")
            
            if pais in df_2019['entity'].values:
                # Obtener características del país
                datos_pais = df_2019[df_2019['entity'] == pais]
                caracteristicas = datos_pais.drop(['year', 'anxiety_disorders', 'eating_disorders', 
                                                'dalys_eating_disorders', 'dalys_anxiety_disorders', 
                                                'entity'], axis=1).values
                
                # Realizar predicción
                prediccion = modelo.predict(caracteristicas)
                resultado = decodificar_prediccion(prediccion[0])
                
                print(f"\nPaís analizado: {pais}")
                print(f"Predicción: {resultado}")
                print(f"Valor de depresión en 2019: {datos_pais['depressive_disorders'].values[0]:.2f}")
            else:
                print(f"\nError: País no encontrado")
            
        except ValueError as ve:
            print(f"\nError: {str(ve)}")
        
    except FileNotFoundError:
        print("Error: No se encontró el archivo del modelo")
    except Exception as e:
        print(f"Error al cargar el modelo: {str(e)}")

if __name__ == "__main__":
    test_modelo_svm()