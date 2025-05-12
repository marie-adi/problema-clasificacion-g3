from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import os

app = FastAPI(title="ML API Clasificación")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/template")

# Cargar el modelo
model_path = "pkl/modelo_svm.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"El modelo no se encuentra en {model_path}")

model = joblib.load(model_path)

@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "resultado": None,
        "probabilidad": None
    })

@app.post("/", response_class=HTMLResponse)
async def predict(
    request: Request,
    entity: str = Form(...)
):
    try:
        # Crear DataFrame de una fila con el nombre del país
        X_new = pd.DataFrame([{
            "Entity": entity
        }])
        
        # Realizar predicción
        pred = model.predict(X_new)[0]
        resultado = f"El país {entity} presenta niveles significativos de depresión" if pred == 1 else f"El país {entity} no presenta niveles significativos de depresión"
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "resultado": resultado,
            "probabilidad": "No disponible",
            "valores": {
                "entity": entity
            }
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Error al procesar el país '{entity}'. Por favor, verifica que el nombre del país sea correcto.",
            "resultado": None,
            "probabilidad": None
        })