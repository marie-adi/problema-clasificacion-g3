from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import joblib
import pandas as pd

app = FastAPI(title="ML API Clasificación")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/template")

# Carga tu modelo de clasificación
model = joblib.load("app/model_clf.pkl")

@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    schizophrenia_disorders: float = Form(...),
    depressive_disorders: float = Form(...),
    bipolar_disorders: float = Form(...),
    anxiety_disorders: float = Form(...),
    eating_disorders: float = Form(...),
):
    # 1) Crear DataFrame de una fila
    X_new = pd.DataFrame([{
        "schizophrenia_disorders": schizophrenia_disorders,
        "depressive_disorders":    depressive_disorders,
        "bipolar_disorders":       bipolar_disorders,
        "anxiety_disorders":       anxiety_disorders,
        "eating_disorders":        eating_disorders,
    }])
    # 2) Predecir probabilidad y etiqueta
    prob = model.predict_proba(X_new)[0,1]
    pred = model.predict(X_new)[0]
    resultado = "Sí" if pred == 1 else "No"

    # 3) Devolver template con resultado y probabilidad
    return templates.TemplateResponse("result.html", {
        "request": request,
        "resultado": resultado,
        "probabilidad": f"{prob:.2%}"
    })
