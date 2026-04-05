import pickle
import numpy as np

SEGMENT_DESCRIPTIONS = {
    "Champions":       "High spenders, frequent buyers, purchased recently. Reward them!",
    "Loyal Customers": "Regular buyers with good spend. Upsell and cross-sell opportunities.",
    "At-Risk":         "Were good customers but haven't purchased recently. Re-engage now.",
    "Lost/Inactive":   "Low spend, infrequent, long since last purchase. Win-back campaigns needed."
}

def predict_segment(recency, frequency, monetary, scaler, kmeans_model, name_map):
    import numpy as np
    val = np.log1p([[recency, frequency, monetary]])
    scaled = scaler.transform(val)
    cluster = kmeans_model.predict(scaled)[0]
    segment = name_map.get(cluster, "Unknown")
    desc = SEGMENT_DESCRIPTIONS.get(segment, "")
    return segment, desc
