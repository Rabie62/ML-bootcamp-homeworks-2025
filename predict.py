import pickle

model_file = 'pipeline_v1.bin'
with open(model_file, 'rb') as f:
    model = pickle.load(f)

client = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

prediction = model.predict_proba([client])[0, 1]
print(f"Probability: {prediction:.3f}")