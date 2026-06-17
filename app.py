import numpy as np
from fertilizer_info import fertilizer_info
from crop_info import crop_info
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import requests
from fastapi.responses import FileResponse

app = FastAPI()

@app.get("/")
def home():
    return FileResponse("home.html")

@app.get("/crop")
def crop():
    return FileResponse("index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = pickle.load(open("crop_model.pkl", "rb"))

API_KEY = "5ef1c00b458c251829e2ad970471eb93"

# 🌱 NPK LEVEL CHECK FUNCTION
def check_npk_levels(n, p, k):

    # Nitrogen
    if n < 50:
        n_status = "Low"
    elif n <= 250:
        n_status = "Balanced"
    else:
        n_status = "High"

    # Phosphorus
    if p < 20:
        p_status = "Low"
    elif p <= 100:
        p_status = "Balanced"
    else:
        p_status = "High"

    # Potassium
    if k < 100:
        k_status = "Low"
    elif k <= 400:
        k_status = "Balanced"
    else:
        k_status = "High"

    return n_status, p_status, k_status


@app.get("/weather/{city}")
def get_weather(city: str):

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(url)
    data = response.json()

    print("API RESPONSE:", data)
    if "main" not in data:
        return {
            "error": "City not found"
        }

    return {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"]
    }

@app.post("/predict")
def predict(data: dict):

    required_fields = ["n", "p", "k", "temperature", "humidity", "ph", "rainfall"]

    # ✅ Check all fields exist
    for field in required_fields:
        if field not in data or data[field] is None:
            return {"error": f"{field} is required"}

    # ✅ Extract values
    n = data.get("n")
    p = data.get("p")
    k = data.get("k")

    # 🌱 Get NPK status
    n_status, p_status, k_status = check_npk_levels(n, p, k)

    # ✅ ML Prediction
    features = [[
        n,
        p,
        k,
        data.get("temperature"),
        data.get("humidity"),
        data.get("ph"),
        data.get("rainfall")
    ]]

    prediction = model.predict(features)[0]
    crop = prediction.lower()

    info = crop_info.get(crop)
    fert = fertilizer_info.get(crop)

    return {
        "crop": crop,
        "n_status": n_status,
        "p_status": p_status,
        "k_status": k_status,
        "image": info["image"],
        "description": info["description"],
        "fertilizer": fert["name"],
        "fertilizer_reason": fert["reason"]
    }
