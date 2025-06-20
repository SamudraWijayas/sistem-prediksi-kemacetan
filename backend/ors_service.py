import requests

ORS_API_KEY = "5b3ce3597851110001cf62481854c320fc606f3c5008d259955e0864a90453ee1bc9344217625be9"

def geocode_address(address):
    if not address:
        raise Exception("Alamat tidak boleh kosong")

    # Jika input berupa koordinat "lat,lon"
    if "," in address and all(part.replace('.', '', 1).replace('-', '', 1).isdigit() for part in address.split(',')):
        lat, lon = map(float, address.split(','))
        return [lon, lat]  # ORS format: [lon, lat]

    url = "https://api.openrouteservice.org/geocode/search"
    headers = {"Authorization": ORS_API_KEY}
    params = {"text": address, "size": 1}

    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    print(f"[DEBUG] Geocode response for '{address}':", data)

    try:
        coords = data["features"][0]["geometry"]["coordinates"]
        print(f"[INFO] Geocoded '{address}' => {coords}")
        return coords  # [lon, lat]
    except (KeyError, IndexError):
        raise Exception(f"Gagal menemukan koordinat untuk lokasi: {address}")

def get_traffic_data_ors(origin, destination):
    start_coords = geocode_address(origin)
    end_coords = geocode_address(destination)

    url = "https://api.openrouteservice.org/v2/directions/driving-car"
    headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}
    body = {
        "coordinates": [start_coords, end_coords]
    }

    response = requests.post(url, headers=headers, json=body)
    data = response.json()

    try:
        summary = data["routes"][0]["summary"]
        distance_km = summary["distance"] / 1000
        duration_sec = summary["duration"]

        # Hitung rata-rata kecepatan (km/jam)
        avg_speed_kmph = (distance_km / (duration_sec / 3600)) if duration_sec > 0 else 0

        # Atur threshold kemacetan berdasarkan kecepatan
        if avg_speed_kmph < 20:
            traffic = "Padat"
        elif avg_speed_kmph < 40:
            traffic = "Sedang"
        else:
            traffic = "Lancar"

        return {
            "distance": distance_km,        # dalam km untuk model ML
            "duration": duration_sec,       # dalam detik
            "traffic_level": traffic
        }
    except (KeyError, IndexError):
        raise Exception("Gagal mengambil data dari ORS: 'routes'")
