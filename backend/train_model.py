import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load data
data = pd.read_csv("backend/data_jalan.csv")

# Fitur & target
X = data[["distance", "duration", "traffic_level"]]
y = data["waktu_tempuh_nyata"]

# Training model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Simpan model
with open("backend/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")


# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# import pickle
# import os

# # Buat folder backend jika belum ada
# os.makedirs("backend", exist_ok=True)

# # Data historis dummy: Jarak, waktu tempuh Google, tingkat kemacetan, dan waktu tempuh riil
# data = pd.DataFrame({
#     'distance': [2.5, 4.0, 5.5, 3.0, 6.0, 3.5, 4.5],
#     'duration': [600, 900, 1200, 750, 1800, 1000, 1500],
#     'traffic_level': [0, 1, 2, 0, 2, 1, 2],  # 0=lancar, 1=sedang, 2=padat
#     'predicted_time': [620, 950, 1300, 780, 2000, 1100, 1700]
# })


# # Fitur dan target
# X = data[['distance', 'duration', 'traffic_level']]
# y = data['predicted_time']

# # Latih model Random Forest
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X, y)

# # Simpan model ke file
# with open('backend/model.pkl', 'wb') as f:
#     pickle.dump(model, f)

# print("✅ Model berhasil dilatih dan disimpan ke backend/model.pkl")
