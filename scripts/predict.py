import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

def load_model(filepath='../models/xgb_best_model.pkl'):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def load_test_data(X_path='../data/test_data/X_test.csv', y_path='../data/test_data/y_test.csv'):
    X_test = pd.read_csv(X_path)
    y_test = pd.read_csv(y_path)
    return X_test, y_test

def predict():
    model = load_model()
    X_test, y_test = load_test_data()
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    
    print("\nPrediction RMSE: ", rmse)
    return preds

if __name__ == "__main__":
    predict()