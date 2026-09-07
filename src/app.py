from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pkl"

model = joblib.load(MODEL_PATH)

class Transaction(BaseModel):
    Time:float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float


app=FastAPI()
@app.get('/')
def home():
    return {'message':'Credit Card Fraud Detection API is running'}

@app.post("/predict")
def predict(transaction: Transaction):

    
    transaction_data = transaction.model_dump()

    
    input_df = pd.DataFrame([transaction_data])

    
    prediction = model.predict(input_df)[0]

    
    probability = model.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "fraud_probability": float(probability),
        "result": (
            "Fraudulent transaction"
            if prediction == 1
            else "Legitimate transaction"
        )
    }
