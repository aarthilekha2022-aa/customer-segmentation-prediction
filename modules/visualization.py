import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os

os.makedirs("outputs", exist_ok=True)

COLORS = ["#4e79a7","#f28e2b","#e15759","#76b7b2"]
PALETTE = dict(zip(["Champions","Loyal Customers","At-Risk","Lost/Inactive"], COLORS))

def plot_elbow(k_range, inertias, silhouettes):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(k_range, inertias, "bo-", linewidth=2)
    ax1.set_title("Elbow Method", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Number of Clusters (k)"); ax1.set_ylabel("Inertia")
    ax1.grid(True, alpha=0.3)
    ax2.plot(k_range, silhouettes, "rs-", linewidth=2)
    ax2.set_title("Silhouette Score", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Number of Clusters (k)"); ax2.set_ylabel("Silhouette Score")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/elbow_silhouette.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_clusters_pca(coords, df):
    fig, ax = plt.subplots(figsize=(9, 6))
    segments = df["Segment"].unique()
    for seg in segments:
        mask = df["Segment"] == seg
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=PALETTE.get(seg, "#888"), label=seg, alpha=0.7, s=55, edgecolors="white", linewidths=0.4)
    ax.set_title("Customer Segments — PCA View", fontsize=14, fontweight="bold")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig("outputs/cluster_pca.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_rfm_boxplots(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, col in zip(axes, ["Recency", "Frequency", "Monetary"]):
        order = ["Champions","Loyal Customers","At-Risk","Lost/Inactive"]
        present = [s for s in order if s in df["Segment"].unique()]
        colors_used = [PALETTE[s] for s in present]
        bp = ax.boxplot([df[df["Segment"]==s][col].values for s in present],
                        patch_artist=True, medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], colors_used):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(present)+1))
        ax.set_xticklabels(present, rotation=20, ha="right", fontsize=9)
        ax.set_title(f"{col} by Segment", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig("outputs/rfm_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_segment_distribution(df):
    counts = df["Segment"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE.get(s,"#888") for s in counts.index],
                  edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+4,
                str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Customer Count per Segment", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count"); ax.set_xlabel("Segment")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/segment_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_rfm_heatmap(df):
    seg_means = (df.groupby("Segment")[["Recency","Frequency","Monetary"]]
                   .mean().round(1))
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(seg_means, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("RFM Mean Values by Segment", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/rfm_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
