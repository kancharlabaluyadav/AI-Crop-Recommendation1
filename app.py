from fertilizer_info import fertilizer_info
from crop_info import crop_info
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = pickle.load(open("crop_model.pkl", "rb"))

API_KEY = "5ef1c00b458c251829e2ad970471eb93"

@app.get("/")
def home():
    return {"message": "Crop Recommendation API"}
@app.get("/weather/{city}")
def get_weather(city: str):

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(url).json()

    temperature = response["main"]["temp"]
    humidity = response["main"]["humidity"]

    return {
        "temperature": temperature,
        "humidity": humidity
    }

@app.post("/predict")
def predict(data: dict):

    features = [[
        data["n"],
        data["p"],
        data["k"],
        data["temperature"],
        data["humidity"],
        data["ph"],
        data["rainfall"]
    ]]

    prediction = model.predict(features)[0]
    crop = prediction.lower()

    info = crop_info.get(crop)
    fert = fertilizer_info.get(crop)

    return {
        "crop": crop,
        "image": info["image"],
        "description": info["description"],
        "fertilizer": fert["name"],
        "fertilizer_reason": fert["reason"]
    }