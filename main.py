from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import joblib
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

modelo = joblib.load(BASE_DIR / "modelo.pkl")

@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")

@app.post("/predict")
def predict(data: dict):
    valores = pd.DataFrame([{
        "Study_Hours_Per_Day": data["Study_Hours_Per_Day"],
        "Sleep_Hours_Per_Day": data["Sleep_Hours_Per_Day"],
        "Extracurricular_Hours_Per_Day": data["Extracurricular_Hours_Per_Day"],
        "Social_Hours_Per_Day": data["Social_Hours_Per_Day"],
        "Stress_Level": data["Stress_Level"]
    }])

    pred = modelo.predict(valores)

    return {"resultado": float(pred.iloc[0])}