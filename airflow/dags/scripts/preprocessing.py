import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    df = df.dropna(subset=["delay"])
    df = df.drop(columns=["flight"])
    
    numeric_features = ['mon', 'dom', 'dow', 'mile', 'depart', 'duration']
    categorical_features = ['carrier', 'org']
    
    numeric_transformer = 'passthrough'
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    X_processed = preprocessor.fit_transform(df)
    y = df["delay"]
    
    feature_names = numeric_features + list(
        preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    )
    
    X = pd.DataFrame(X_processed, columns=feature_names)
    return X, y