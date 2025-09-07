from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import pickle

#load scaler

with open("wine_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load model
with open("rf_classifier_model.pkl", "rb") as f:
    model = pickle.load(f)




class WineQuality(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float



app = FastAPI()


@app.post("/predict")
def predict_quality(wine: WineQuality):
    data = np.array([[wine.fixed_acidity, wine.volatile_acidity, wine.citric_acid, wine.residual_sugar,
                      wine.chlorides, wine.free_sulfur_dioxide, wine.total_sulfur_dioxide, wine.density,
                      wine.pH, wine.sulphates, wine.alcohol]])
    
    # Scale the input data
    data_scaled = scaler.transform(data)
    
    # Make prediction
    prediction = model.predict(data_scaled)
    
    return {"predicted_quality": int(prediction[0])}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)