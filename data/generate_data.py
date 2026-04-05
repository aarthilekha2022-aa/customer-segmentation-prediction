import pandas as pd
import numpy as np

np.random.seed(42)
n = 500

customer_ids = [f"CUST{str(i).zfill(4)}" for i in range(1, n+1)]

recency    = np.concatenate([np.random.randint(1,30,100), np.random.randint(30,90,150),
                              np.random.randint(90,200,150), np.random.randint(200,365,100)])
frequency  = np.concatenate([np.random.randint(15,50,100), np.random.randint(8,20,150),
                              np.random.randint(3,10,150),  np.random.randint(1,4,100)])
monetary   = np.concatenate([np.random.randint(5000,20000,100), np.random.randint(1500,6000,150),
                              np.random.randint(400,2000,150),   np.random.randint(50,500,100)])

idx = np.random.permutation(n)
recency, frequency, monetary = recency[idx], frequency[idx], monetary[idx]

df = pd.DataFrame({"CustomerID": customer_ids,
                   "Recency": recency,
                   "Frequency": frequency,
                   "Monetary": monetary})
df.to_csv("data/rfm_data.csv", index=False)
print(f"Dataset saved — {len(df)} rows")
print(df.describe())
