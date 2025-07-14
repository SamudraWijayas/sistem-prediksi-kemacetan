import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import mse
from ors_service import get_traffic_data_ors as get_traffic_data

# Load model dan scaler
model = load_model("backend/model_lstm.h5", custom_objects={'mse': mse})
scaler_X = joblib.load("backend/scaler_X.pkl")
scaler_y = joblib.load("backend/scaler_y.pkl")

st.title("Sistem Prediksi Kemacetan & Waktu Tempuh ke Kampus Darmajaya")

origin = st.text_input("Lokasi Awal (Origin)")
destination = st.text_input("Tujuan (Destination)")

if st.button("Prediksi"):
    try:
        traffic = get_traffic_data(origin, destination)

        traffic_map = {"Lancar": 0, "Sedang": 1, "Padat": 2}
        traffic_code = traffic_map.get(traffic["traffic_level"], 0)

        features = np.array([[traffic["distance"], traffic["duration"], traffic_code]])
        features_scaled = scaler_X.transform(features).reshape((1, 1, 3))

        pred_scaled = model.predict(features_scaled)
        prediction = scaler_y.inverse_transform(pred_scaled)[0][0]
        prediction = round(prediction, 2)

        st.success("Hasil Prediksi:")
        st.write(f"**Jarak**: {round(traffic['distance'] * 1000, 2)} meter")
        st.write(f"**Durasi Aktual**: {round(traffic['duration'] / 60, 2)} menit")
        st.write(f"**Tingkat Kemacetan**: {traffic['traffic_level']}")
        st.write(f"**Prediksi Durasi Tempuh**: {prediction} menit")

    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {str(e)}")
