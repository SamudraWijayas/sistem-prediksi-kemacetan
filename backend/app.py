import streamlit as st
from streamlit_folium import st_folium
import folium
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import mse
from ors_service import get_traffic_data_ors as get_traffic_data

# Load model dan scaler
model = load_model("backend/model_lstm.h5", custom_objects={'mse': mse})
scaler_X = joblib.load("backend/scaler_X.pkl")
scaler_y = joblib.load("backend/scaler_y.pkl")

st.set_page_config(page_title="Prediksi Waktu Tempuh", layout="centered")

st.title("📍 Prediksi Waktu Tempuh ke Kampus Darmajaya")
st.markdown("Klik pada peta untuk memilih lokasi asal.")

# Default koordinat Darmajaya
darmajaya_coords = [-5.37717, 105.24962]

# Buat peta interaktif
m = folium.Map(location=darmajaya_coords, zoom_start=13)
folium.Marker(darmajaya_coords, tooltip="Darmajaya 🎓", icon=folium.Icon(color="green")).add_to(m)

# Komponen peta di Streamlit
map_data = st_folium(m, height=400, width=700)

# Ambil koordinat dari klik peta
origin_coords = None
if map_data and map_data.get("last_clicked"):
    origin_coords = map_data["last_clicked"]
    st.success(f"Lokasi asal dipilih: {origin_coords['lat']}, {origin_coords['lng']}")

# Form prediksi
if origin_coords and st.button("🔍 Hitung Prediksi Waktu"):
    try:
        origin = f"{origin_coords['lat']},{origin_coords['lng']}"
        destination = f"{darmajaya_coords[0]},{darmajaya_coords[1]}"

        traffic = get_traffic_data(origin, destination)

        traffic_map = {"Lancar": 0, "Sedang": 1, "Padat": 2}
        traffic_code = traffic_map.get(traffic["traffic_level"], 0)

        features = np.array([[traffic["distance"], traffic["duration"], traffic_code]])
        features_scaled = scaler_X.transform(features).reshape((1, 1, 3))
        pred_scaled = model.predict(features_scaled)
        prediction = scaler_y.inverse_transform(pred_scaled)[0][0]
        prediction = round(prediction, 2)

        st.markdown("### 📊 Hasil Prediksi")
        st.write(f"**Jarak:** {round(traffic['distance'] * 1000, 2)} meter")
        st.write(f"**Durasi dari ORS:** {round(traffic['duration'] / 60, 2)} menit")
        st.write(f"**Kondisi Lalu Lintas:** {traffic['traffic_level']}")
        st.success(f"⏱️ Prediksi Waktu Tempuh (ML): **{prediction} menit**")

    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {str(e)}")
else:
    st.info("Silakan pilih lokasi asal pada peta terlebih dahulu.")

