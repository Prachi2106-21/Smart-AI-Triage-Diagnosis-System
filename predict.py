import pandas as pd
from joblib import load

# Load trained model and label binarizer
model = load('analysis/artifacts/diagnostic_model.pkl')
mlb = load('analysis/artifacts/label_binarizer.pkl')

def predict_diagnosis(input_dict):
    # Convert input to DataFrame
    df = pd.DataFrame([input_dict])
    
    # Predict
    pred = model.predict(df)
    labels = mlb.inverse_transform(pred)
    
    return labels[0]  # list of predicted diagnoses

# 🧪 Test run
if __name__ == "__main__":
    test_patient = {
        'fever': 1,
        'cough': 1,
        'chest_pain': 1,
        'wbc': 18000,
        'crp': 120,
        'metformin': 0,
        'lisinopril': 0
    }
    
    results = predict_diagnosis(test_patient)
    print("🧠 Predicted Diagnoses:", results)