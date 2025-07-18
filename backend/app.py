from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import mse
from ors_service import get_traffic_data_ors as get_traffic_data

app = Flask(__name__)

# Load LSTM model dan scaler
model = load_model("backend/model_lstm.h5", custom_objects={'mse': mse})
scaler_X = joblib.load("backend/scaler_X.pkl")
scaler_y = joblib.load("backend/scaler_y.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        origin = request.form.get("origin")
        destination = request.form.get("destination")

        try:
            traffic = get_traffic_data(origin, destination)
            traffic_map = {"Lancar": 0, "Sedang": 1, "Padat": 2}
            traffic_code = traffic_map.get(traffic["traffic_level"], 0)

            features = np.array([[traffic["distance"], traffic["duration"], traffic_code]])
            features_scaled = scaler_X.transform(features).reshape((1, 1, 3))

            pred_scaled = model.predict(features_scaled)
            prediction = scaler_y.inverse_transform(pred_scaled)[0][0]
            prediction = round(prediction, 2)

            result = {
                "distance": round(traffic["distance"] * 1000, 2),
                "duration": round(traffic["duration"] / 60, 2),
                "traffic": traffic["traffic_level"],
                "predicted_duration": prediction
            }
            print("Debug Result:", result)

        except Exception as e:
            error = str(e)

    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    app.run(debug=True)

# from flask import Flask, render_template, request
# import pickle
# import pandas as pd
# from ors_service import get_traffic_data_ors as get_traffic_data

# app = Flask(__name__)

# # Load model
# with open("backend/model.pkl", "rb") as f:
#     model = pickle.load(f)

# @app.route("/", methods=["GET", "POST"])
# def index():
#     result = None
#     error = None

#     if request.method == "POST":
#         origin = request.form.get("origin")
#         destination = request.form.get("destination")

#         try:
#             traffic = get_traffic_data(origin, destination)

#             traffic_map = {"Lancar": 0, "Sedang": 1, "Padat": 2}
#             traffic_code = traffic_map.get(traffic["traffic_level"], 0)

#             features = pd.DataFrame([{
#                 "distance": traffic["distance"],  # tetap dalam km untuk model
#                 "duration": traffic["duration"],  # detik
#                 "traffic_level": traffic_code
#             }])

#             prediction = round(model.predict(features)[0], 2)

#             result = {
#                 "distance": round(traffic["distance"] * 1000, 2),  # km ke meter
#                 "duration": round(traffic["duration"] / 60, 2),    # detik ke menit
#                 "traffic": traffic["traffic_level"],
#                 "predicted_duration": prediction
#             }
#             print("Debug Result:", result)

#         except Exception as e:
#             error = str(e)

#     return render_template("index.html", result=result, error=error)

# if __name__ == "__main__":
#     app.run(debug=True)


