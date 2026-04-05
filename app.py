import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.preprocessing import load_data, preprocess, get_summary
from modules.clustering     import find_optimal_k, train_kmeans, apply_pca, assign_segment_names
from modules.prediction     import predict_segment, SEGMENT_DESCRIPTIONS
from modules.visualization  import (plot_elbow, plot_clusters_pca, plot_rfm_boxplots,
                                    plot_segment_distribution, plot_rfm_heatmap)

st.set_page_config(page_title="Customer Segmentation", page_icon="📊", layout="wide")

st.markdown("""
<style>
.metric-card{background:#f8f9fa;border-radius:10px;padding:16px;text-align:center;border:1px solid #e0e0e0}
.seg-badge{padding:6px 14px;border-radius:20px;font-weight:600;font-size:15px;display:inline-block}
</style>""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x60?text=BigSiBucks", width=200)
    st.title("⚙️ Settings")
    n_clusters = st.slider("Number of Clusters (k)", 2, 8, 4)
    uploaded   = st.file_uploader("Upload your RFM CSV", type="csv")
    st.markdown("---")
    st.caption("BigSiBucks Innovation Pvt Ltd")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def get_data(uploaded):
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        df = load_data("data/rfm_data.csv")
    return df

df_raw = get_data(uploaded)
df, X_scaled, scaler, features = preprocess(df_raw)
km, labels, sil, db            = train_kmeans(X_scaled, n_clusters)
df, name_map                   = assign_segment_names(df, labels)
coords, _                      = apply_pca(X_scaled)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Customer Segmentation Prediction")
st.caption("RFM Analysis · K-Means Clustering · BigSiBucks Innovation Pvt Ltd")
st.markdown("---")

# ── KPI Row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
summary = get_summary(df_raw)
c1.metric("Total Customers",  summary["total_customers"])
c2.metric("Avg Recency",      f"{summary['avg_recency']} days")
c3.metric("Avg Frequency",    f"{summary['avg_frequency']} orders")
c4.metric("Avg Monetary",     f"₹{summary['avg_monetary']:,}")
c5.metric("Silhouette Score", sil)

st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Segments", "📈 RFM Analysis", "🔍 Elbow Method", "🎯 Predict"])

with tab1:
    st.subheader("Customer Segments (PCA View)")
    plot_clusters_pca(coords, df)
    st.image("outputs/cluster_pca.png", use_column_width=True)

    st.subheader("Segment Distribution")
    plot_segment_distribution(df)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("outputs/segment_distribution.png", use_column_width=True)
    with col2:
        seg_counts = df["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        seg_counts["% Share"] = (seg_counts["Count"] / len(df) * 100).round(1)
        st.dataframe(seg_counts, use_container_width=True)

with tab2:
    st.subheader("RFM Distribution by Segment")
    plot_rfm_boxplots(df)
    st.image("outputs/rfm_boxplots.png", use_column_width=True)
    st.subheader("RFM Heatmap (Segment Averages)")
    plot_rfm_heatmap(df)
    st.image("outputs/rfm_heatmap.png", use_column_width=True)

with tab3:
    st.subheader("Elbow Method & Silhouette Score")
    k_range, inertias, silhouettes = find_optimal_k(X_scaled)
    plot_elbow(k_range, inertias, silhouettes)
    st.image("outputs/elbow_silhouette.png", use_column_width=True)
    st.info(f"Best k by Silhouette = **{k_range[silhouettes.index(max(silhouettes))]}** | "
            f"Davies-Bouldin Index = **{db}**")

with tab4:
    st.subheader("🎯 Predict Segment for New Customer")
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        recency   = col1.number_input("Recency (days since last purchase)", 1, 365, 30)
        frequency = col2.number_input("Frequency (number of orders)", 1, 100, 10)
        monetary  = col3.number_input("Monetary (total spend ₹)", 100, 100000, 3000)
        submitted = st.form_submit_button("🔍 Predict Segment")

    if submitted:
        segment, desc = predict_segment(recency, frequency, monetary, scaler,
                                        pickle.load(open("models/kmeans_model.pkl","rb")), name_map)
        COLOR_MAP = {"Champions":"#2ecc71","Loyal Customers":"#3498db",
                     "At-Risk":"#e67e22","Lost/Inactive":"#e74c3c"}
        color = COLOR_MAP.get(segment, "#888")
        st.markdown(f"""
        <div style='background:{color}22;border-left:5px solid {color};
                    border-radius:8px;padding:18px;margin-top:12px'>
          <h3 style='color:{color};margin:0'>Segment: {segment}</h3>
          <p style='margin:8px 0 0;font-size:15px'>{desc}</p>
        </div>""", unsafe_allow_html=True)

# ── Data Table ────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📋 View Segmented Customer Data"):
    st.dataframe(df[["CustomerID","Recency","Frequency","Monetary","Segment"]],
                 use_container_width=True)
    csv = df.to_csv(index=False).encode()
    st.download_button("⬇️ Download CSV", csv, "segmented_customers.csv", "text/csv")
