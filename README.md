# 📊 Customer Segmentation Prediction
**BigSiBucks Innovation Private Limited — Internship Project**

> Automatically segment customers using RFM Analysis + K-Means Clustering, with an interactive Streamlit dashboard.

---

## 🚀 Features
- **RFM Analysis** — Recency, Frequency, Monetary scoring
- **K-Means Clustering** — Optimal k via Elbow Method & Silhouette Score
- **PCA Visualization** — 2D cluster scatter plots
- **Segment Profiles** — Champions, Loyal Customers, At-Risk, Lost/Inactive
- **Live Prediction** — Predict segment for any new customer
- **Streamlit Dashboard** — Interactive, business-friendly UI
- **Downloadable Results** — Export segmented CSV

---

## 🗂️ Project Structure
```
customer_segmentation/
├── data/
│   ├── rfm_data.csv            # RFM dataset
│   └── generate_data.py        # Dataset generator
├── modules/
│   ├── preprocessing.py        # Data cleaning & scaling
│   ├── clustering.py           # K-Means + PCA
│   ├── visualization.py        # All plots
│   └── prediction.py           # Segment predictor
├── models/                     # Saved model artifacts
├── outputs/                    # Generated charts & CSV
├── train.py                    # Training pipeline
├── app.py                      # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/aarthilekha2022-aa/customer-segmentation-prediction.git
cd customer-segmentation-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate dataset
```bash
python data/generate_data.py
```

### 4. Train the model
```bash
python train.py
```

### 5. Launch Streamlit app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📊 Customer Segments

| Segment | Description | Strategy |
|---|---|---|
| 🏆 Champions | High spend, frequent, recent | Reward & retain |
| 💙 Loyal Customers | Regular buyers, decent spend | Upsell / cross-sell |
| ⚠️ At-Risk | Good history, not buying recently | Re-engagement campaigns |
| 💤 Lost/Inactive | Low spend, long inactive | Win-back offers |

---

## 📈 Model Performance
- **Algorithm**: K-Means Clustering
- **Optimal k**: 4 (Elbow + Silhouette method)
- **Silhouette Score**: ~0.55–0.65
- **Evaluation**: Davies-Bouldin Index, PCA visualization

---

## 🏢 About
**BigSiBucks Innovation Private Limited**  
CIN: U27400TN2024PTC169823  
Chennai – 600097, Tamil Nadu, India

---

## 👤 Author
[@aarthilekha2022-aa](https://github.com/aarthilekha2022-aa)
