## TE HAS SENTIDO MAL MENTALMENTE ULTIMAMENTE ? QUE TAL SI TE DIGO PORQUE ES EL PAIS EN DONDE VIVES , QUIERES SABERLO ? ESTE PROYECTO TE PUEDE INTERESAR 

## Acerca de este proyecto 
el origen de la idea surgio sobre la importancia de la salud en general y nos queriamos enfocar en eso hasta que llegamos a la salud mental y la importancia de la misma sabes que hay estudios que indican de que si sufres de depresion en general es algo sumamente malo pero hay informas que indican que te acorta la expectativa de vida. Partiendo de esto sacamos un dataset de kaggle con el que llegamos a mental healh un dataset que se enfonca tanto en la depresion como en otras enfermedades mentales y cambiamos la forma incluso de visualizar el proyecto porque nos muestra los paises mas afectados por salud mental 

##  🔍 Main Features  
✅ Complete EDA with visualizations to understand variable relationships.  
✅ Trained  Gradient Boosting model to predict used car prices.  
✅ Fast api for deploment.  
✅ Well-structured project by functionality.  

##  🐞 Current Issues  
❌ hay ciertos problemas con el eda por muchos csv y no todos contienen los mismos datos.

# EDA - Problema de Clasificación

## Descripción del Proyecto

Este proyecto consiste en un análisis exploratorio de datos (EDA) enfocado en la variable objetivo `depressive_disorders`. El objetivo es comprender la estructura del dataset, identificar patrones significativos y preparar los datos para su posterior modelado predictivo utilizando FastAPI, HTML y Jinja2.

## Estructura del Notebook

El notebook está organizado en las siguientes secciones:

1. **Configuración Inicial y Preprocesamiento:**

   * Importación de librerías esenciales: `Pandas`, `Numpy`, `Matplotlib`, `Seaborn`, `Scikit-Learn`, `FastAPI`, `Jinja2`.
   * Carga del dataset en formato CSV y configuración del entorno de trabajo.
   * Preprocesamiento de datos, incluyendo limpieza y conversión de variables categóricas a numéricas.

2. **Análisis Univariante:**

   * Análisis de la distribución de la variable objetivo `depressive_disorders` mediante histogramas y gráficos KDE.
   * Identificación de una distribución multimodal con tres picos principales (21-22%, 27-29% y 38%).

3. **Análisis Bivariante:**

   * Evaluación de correlaciones entre la variable objetivo y otras variables del dataset.
   * Visualización de relaciones a través de gráficos de dispersión y mapas de calor.

4. **Integración con FastAPI y HTML:**

   * Implementación de endpoints para servir los análisis y visualizaciones.
   * Uso de Jinja2 para renderizar plantillas HTML dinámicas.

## Requisitos

* Python 3.9 o superior
* Bibliotecas necesarias:

  * pandas
  * numpy
  * matplotlib
  * seaborn
  * scikit-learn
  * fastapi
  * jinja2

## Ejecución del Proyecto

1. Clonar el repositorio:

   ```bash
   git clone <URL_REPOSITORIO>
   ```

2. Instalar las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

3. Ejecutar el servidor FastAPI:

   ```bash
   uvicorn main:app --reload
   ```

4.Instrucciones de uso de esta API:

## 🛠️ Instalación & Ejecución

> **Requisito:** Python 3.11

### 1️⃣  Crear y activar entorno virtual
``` textplain
python3.11 -m venv .venv
```
 macOS/Linux
```textplain
source .venv/bin/activate
``` 
 Windows
 ```
 .venv\Scripts\activate
```   

### 2️⃣  Instalar dependencias
```textplain
pip install -r requirements.txt
```

> [!TIP]
> Con `pip list` puedes visualizar todas las dependencias descargadas.

### 3️⃣  Ejecutar fastAPI
```textplain
fastapi dev app/main.py
````

## Autor

* [Mariela Adimari](https://github.com/marie-adi)
* [Abigail Masapanta](https://github.com/abbyenredes)
* [Orlando Alcalá](https://github.com/odar1997)
* [Jorge Mateo Reyes](https://github.com/Jorgeluuu)

