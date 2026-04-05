import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import pickle, os

def find_optimal_k(X_scaled, k_range=range(2, 10)):
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    return list(k_range), inertias, silhouettes

def train_kmeans(X_scaled, n_clusters=4):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil   = round(silhouette_score(X_scaled, labels), 4)
    db    = round(davies_bouldin_score(X_scaled, labels), 4)
    os.makedirs("models", exist_ok=True)
    with open("models/kmeans_model.pkl", "wb") as f:
        pickle.dump(km, f)
    return km, labels, sil, db

def apply_pca(X_scaled):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    return coords, pca

SEGMENT_NAMES = {0: "Champions", 1: "Loyal Customers",
                 2: "At-Risk",   3: "Lost/Inactive"}

def assign_segment_names(df, labels):
    # Sort clusters by Monetary mean to assign meaningful names
    df = df.copy()
    df["Cluster"] = labels
    order = (df.groupby("Cluster")["Monetary"].mean()
               .sort_values(ascending=False)
               .index.tolist())
    name_map = {cluster: list(SEGMENT_NAMES.values())[i]
                for i, cluster in enumerate(order)}
    df["Segment"] = df["Cluster"].map(name_map)
    return df, name_map
