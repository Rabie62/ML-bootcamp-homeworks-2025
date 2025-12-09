from fastapi import FastAPI
from pydantic import BaseModel
import pickle

class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

with open("pipeline_v1.bin", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

@app.post("/predict")
def predict(lead: Lead):
    data = lead.dict()
    pred = model.predict_proba([data])[0, 1]
    return {"conversion_probability": round(pred, 3)}