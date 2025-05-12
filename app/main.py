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
model = joblib.load("app/modelo_svm.pkl")

@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    entity: str = Form(...),

):
    # 1) Crear DataFrame de una fila
    X_new = pd.DataFrame([{
        "entity": entity,
       
    }])
    # 2) Predecir probabilidad y etiqueta
    pred = model.predict(X_new)[0]
    resultado = "Sí" if pred == 1 else "No"
    print("-------------",resultado)

    # 3) Devolver template con resultado y probabilidad
    return templates.TemplateResponse("index.html", {
        "request": request,
        "resultado": resultado,
        "probabilidad": f"{pred:.2%}"
    })
