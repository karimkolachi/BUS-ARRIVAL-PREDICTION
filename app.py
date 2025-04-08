from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

# Load model and scaler
model = load_model("bus_arrival_lstm_updated.h5", compile=False)
scaler = joblib.load("scaler.pkl")  # Save your scaler using joblib.dump(scaler, 'scaler.pkl')

app = FastAPI()

class RequestData(BaseModel):
    distance_to_stop: float
    traffic_level: str
    weather_condition: str
    current_speed: float

@app.post("/predict")
def predict(data: RequestData):
    traffic_encoding = {"Low": 0, "Moderate": 1, "High": 2}
    weather_encoding = {"Clear": 0, "Rainy": 1, "Foggy": 2, "Snowy": 3}

    traffic_encoded = traffic_encoding.get(data.traffic_level, 1)
    weather_encoded = weather_encoding.get(data.weather_condition, 0)

    input_features = np.array([[data.distance_to_stop, traffic_encoded,
                                weather_encoded, data.current_speed]])
    input_scaled = scaler.transform(input_features)
    input_scaled = input_scaled.reshape((1, 1, input_scaled.shape[1]))

    prediction = model.predict(input_scaled)[0][0]
    return {"Predicted_Arrival_Time_Minutes": round(float(prediction), 2)}
