import requests
import json

url = "https://power.larc.nasa.gov/api/temporal/daily/point" 
params = {
        "latitude": 39.47391, #Valencia, España
        "longitude": -0.37966,
        "start": 20000101,
        "end": 20250101,
        "parameters": "ALLSKY_SFC_SW_DWN,T2M,WS2M,PRECTOTCORR", # ALLSKY_SFC_SW_DWN es la radiación real que llega a la superficie descontando la nubosidad, T2M es al temperatura del aire, WS2M es la velocidad del viento y PRETOTCORR es la Precipitación
        "community": "RE",
        "format": "JSON"
}

response = requests.get(url=url, params=params)
data = response.json()

with open('data/initial_data.json', 'w') as file:
    json.dump(data, file, indent=3)