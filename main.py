from typing import Union

from fastapi import FastAPI

app = FastAPI(
    title="Análisis salud mental",
    description="API para consultar y analizar datos de salud mental",
    version="1.0.0")

# @app.get("/")
# def read_root():
  #  return {"Hello": "World"}
# Rutas principales
@app.get("/") 
def welcome():
    return {'message': "Bienvenido a la API de Análisis de Datos de salud mental"}



@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

