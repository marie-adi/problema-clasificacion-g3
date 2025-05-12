import pandas as pd
import joblib
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Inicializa la app
app = FastAPI(title="ML API Clasificación")

# Monta los archivos estáticos y las plantillas
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/template")

# Carga del modelo y del CSV al iniciar la aplicación
model = joblib.load("/home/abby/problema-clasificacion-g3/app/modelo_svm.pkl")
df = pd.read_csv("/home/abby/problema-clasificacion-g3/Data/df_final.csv")
df_2019 = df[df['year'] == 2019].copy()

# Diccionario para decodificar la predicción
decodificacion = {
    0: "No presenta niveles significativos de depresión",
    1: "Presenta niveles significativos de depresión"
}

@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    # Pasamos la lista ordenada de países para sugerencias
    paises = sorted(df_2019['entity'].unique())
    return templates.TemplateResponse("index.html", {"request": request, "paises": paises})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, entity: str = Form(...)):
    # Buscamos el país ignorando mayúsculas/minúsculas
    row = df_2019[df_2019['entity'].str.lower() == entity.strip().lower()]
    if row.empty:
        error = f"No se encontró el país '{entity}'."
        paises = sorted(df_2019['entity'].unique())
        return templates.TemplateResponse("index.html", {"request": request, "error": error, "paises": paises})

    # Extraemos las características para predecir
    X_new = row.drop([
        'year', 'anxiety_disorders', 'eating_disorders',
        'dalys_eating_disorders', 'dalys_anxiety_disorders', 'entity'
    ], axis=1)

    # Predicción y decodificación
    pred = model.predict(X_new)[0]
    resultado = decodificacion.get(pred, f"Predicción desconocida (valor: {pred})")

    # Si el modelo soporta probabilidades, las obtenemos
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(X_new)[0][pred]
        prob_str = f"{prob:.2%}"
    else:
        prob_str = None

    # Valor real de depresión en 2019
    valor_real = f"{row['depressive_disorders'].values[0]:.2f}"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "resultado": resultado,
        "probabilidad": prob_str,
        "valor_real": valor_real,
        "paises": sorted(df_2019['entity'].unique())
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
