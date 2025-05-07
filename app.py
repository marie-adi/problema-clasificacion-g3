from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load("archivo.pkl")

app = FastAPI(
    title="Análisis salud mental",
    description="API para predecir la salud mental",
    version="1.0.0")

#
class MentalHealthInput(BaseModel):
   variable1: float
   variable2: float

class MentalHealtPrediction(BaseModel):
    predicted_class: int
    predicted_class_name: str


@app.post("/predict", response_model=MentalHealtPrediction)
def predict(data: MentalHealthInput):
    # Convert the input data to a numpy array
    input_data = np.array(
        [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    )

    # Make a prediction
    predicted_class = model.predict(input_data)[0]
    predicted_class_name = variable1().target_names[predicted_class]

    return MentalHealtPrediction(
        predicted_class=predicted_class, predicted_class_name=predicted_class_name
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)