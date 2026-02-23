import pandas as pd
import joblib
import json
from sklearn.metrics import mean_squared_error, r2_score
import os

def evaluate_model():
    X_test = pd.read_csv('data/processed_data/X_test_scaled.csv')
    y_test = pd.read_csv('data/processed_data/y_test.csv').values.ravel()
    model = joblib.load('models/gbr_model.pkl')
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    metrics = {"mse": mse, "r2": r2}
    
    os.makedirs('metrics', exist_ok=True)
    with open('metrics/scores.json', 'w') as f:
        json.dump(metrics, f)
        
    pd.DataFrame(predictions, columns=['prediction']).to_csv('data/prediction.csv', index=False)

if __name__ == "__main__":
    evaluate_model()