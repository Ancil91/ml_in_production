from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(title="California Housing Price Prediction")

try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    model_loaded = True
except Exception:
    model = None
    model_loaded = False
    

class HousingData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float
    
@app.get("/")
def home():
    return FileResponse("index.html")
    
    
@app.get("/health")
def health_check():
    return{
        "status":"ok",
        "model_loaded": model_loaded
    }
    
@app.post("/predict")
def predict(data: HousingData):
    if model is None:
        return{
            "error": "Model not found."
        }
        
    features = np.array([[
        data.MedInc,
        data.HouseAge,
        data.AveRooms,
        data.AveBedrms,
        data.Population,
        data.AveOccup,
        data.Latitude,
        data.Longitude
    ]])

    prediction = model.predict(features)
    
    return {
        "predicted_house_value": float(prediction[0])
    }
    
