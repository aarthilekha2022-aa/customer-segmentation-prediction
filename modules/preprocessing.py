import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def load_data(filepath="data/rfm_data.csv"):
    df = pd.read_csv(filepath)
    return df

def preprocess(df):
    df = df.dropna()
    df = df.drop_duplicates(subset="CustomerID")

    # Log-transform to reduce skew
    df["Recency_log"]   = np.log1p(df["Recency"])
    df["Frequency_log"] = np.log1p(df["Frequency"])
    df["Monetary_log"]  = np.log1p(df["Monetary"])

    features = ["Recency_log", "Frequency_log", "Monetary_log"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    return df, X_scaled, scaler, features

def get_summary(df):
    return {
        "total_customers": len(df),
        "avg_recency":     round(df["Recency"].mean(), 1),
        "avg_frequency":   round(df["Frequency"].mean(), 1),
        "avg_monetary":    round(df["Monetary"].mean(), 2),
        "missing_values":  df.isnull().sum().sum()
    }
