import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os

def run_grid_search():
    X_train = pd.read_csv('data/processed_data/X_train_scaled.csv')
    y_train = pd.read_csv('data/processed_data/y_train.csv').values.ravel()
    
    model = GradientBoostingRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }
    
    grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(grid.best_params_, 'models/best_params.pkl')

if __name__ == "__main__":
    run_grid_search()