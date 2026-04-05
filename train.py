import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.preprocessing  import load_data, preprocess, get_summary
from modules.clustering      import find_optimal_k, train_kmeans, apply_pca, assign_segment_names
from modules.visualization   import (plot_elbow, plot_clusters_pca, plot_rfm_boxplots,
                                     plot_segment_distribution, plot_rfm_heatmap)

import pickle, os

def main():
    print("=" * 55)
    print("  Customer Segmentation Prediction — Training Pipeline")
    print("=" * 55)

    # 1. Load & preprocess
    print("\n[1] Loading dataset...")
    df = load_data("data/rfm_data.csv")
    summary = get_summary(df)
    print(f"    Total customers : {summary['total_customers']}")
    print(f"    Avg Recency     : {summary['avg_recency']} days")
    print(f"    Avg Frequency   : {summary['avg_frequency']} orders")
    print(f"    Avg Monetary    : ₹{summary['avg_monetary']}")
    print(f"    Missing values  : {summary['missing_values']}")

    df, X_scaled, scaler, features = preprocess(df)
    print(f"\n[2] Preprocessing complete — features: {features}")

    # 2. Optimal k
    print("\n[3] Finding optimal k (2–9)...")
    k_range, inertias, silhouettes = find_optimal_k(X_scaled)
    plot_elbow(k_range, inertias, silhouettes)
    best_k = k_range[silhouettes.index(max(silhouettes))]
    print(f"    Best k by silhouette = {best_k}")

    # 3. Train
    print(f"\n[4] Training K-Means (k={best_k})...")
    km, labels, sil, db = train_kmeans(X_scaled, n_clusters=best_k)
    print(f"    Silhouette Score    : {sil}")
    print(f"    Davies-Bouldin Index: {db}")

    # 4. Assign segment names
    df, name_map = assign_segment_names(df, labels)
    print(f"\n[5] Segment distribution:")
    for seg, cnt in df["Segment"].value_counts().items():
        print(f"    {seg:20s}: {cnt} customers")

    # 5. PCA & plots
    print("\n[6] Generating visualizations...")
    coords, pca = apply_pca(X_scaled)
    plot_clusters_pca(coords, df)
    plot_rfm_boxplots(df)
    plot_segment_distribution(df)
    plot_rfm_heatmap(df)
    print("    Saved to outputs/")

    # 6. Save artifacts
    os.makedirs("models", exist_ok=True)
    with open("models/scaler.pkl",   "wb") as f: pickle.dump(scaler, f)
    with open("models/name_map.pkl", "wb") as f: pickle.dump(name_map, f)
    df.to_csv("outputs/segmented_customers.csv", index=False)
    print("\n[7] Model artifacts saved to models/")
    print("    Segmented data   saved to outputs/segmented_customers.csv")

    print("\n" + "=" * 55)
    print("  Training complete! Run: streamlit run app.py")
    print("=" * 55)

if __name__ == "__main__":
    main()
