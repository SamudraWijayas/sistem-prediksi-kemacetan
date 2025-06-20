import requests

def get_traffic_data(origin, destination, api_key, departure_time_unix):
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": origin,
        "destinations": destination,
        "departure_time": departure_time_unix,
        "key": api_key
    }

    response = requests.get(url, params=params)
    data = response.json()

    try:
        element = data['rows'][0]['elements'][0]
        if element['status'] != 'OK':
            raise ValueError("Data tidak valid dari Google Maps API")

        distance_km = element['distance']['value'] / 1000  # meters to km
        duration_val = element['duration_in_traffic']['value']  # detik
        duration_text = element['duration_in_traffic']['text']

        # Tentukan tingkat kemacetan
        if duration_val > 1800:
            traffic = "Padat"
        elif duration_val > 900:
            traffic = "Sedang"
        else:
            traffic = "Lancar"

        return {
            "distance": distance_km,
            "duration": duration_val,
            "duration_text": duration_text,
            "traffic_level": traffic
        }
    except (KeyError, IndexError, ValueError) as e:
        print("Error saat mengambil data dari API:", e)
        raise Exception("Gagal mengambil data lalu lintas.")
