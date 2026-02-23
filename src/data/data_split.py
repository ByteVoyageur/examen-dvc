import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_data():
    df = pd.read_csv('data/raw_data/raw.csv')
    
    if df.iloc[:, 0].dtype == 'object' or '-' in str(df.iloc[0, 0]):
        df = df.iloc[:, 1:]
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    output_dir = 'data/processed_data'
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False)

if __name__ == "__main__":
    split_data()