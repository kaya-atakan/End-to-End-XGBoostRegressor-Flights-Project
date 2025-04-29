import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from preprocessing import load_data, preprocess_data

def save_model(model, filename='../models/xgb_best_model.pkl'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def train():
    df = load_data('../data/raw_data/flights_raw.csv')
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_test.to_csv('../data/test_data/X_test.csv', index=False)
    y_test.to_csv('../data/test_data/y_test.csv', index=False)

    gbm_param_grid = {
        'colsample_bytree': [0.3, 0.7],
        'n_estimators': [50, 100],
        'max_depth': [2, 5],
        'learning_rate': [0.01, 0.1, 0.3],
        'reg_alpha': [0, 0.1, 1]
    }

    gbm = XGBRegressor(random_state=42)

    grid_mse = GridSearchCV(
        estimator=gbm,
        param_grid=gbm_param_grid, 
        scoring='neg_mean_squared_error',
        cv=4,
        verbose=1,
        n_jobs=-1
    )

    grid_mse.fit(X_train, y_train)

    best_model = grid_mse.best_estimator_
   

    print("\nBest Parameters: ", grid_mse.best_params_)

    return best_model

if __name__ == "__main__":
    best_model = train()
    save_model(best_model, '../models/xgb_best_model.pkl')