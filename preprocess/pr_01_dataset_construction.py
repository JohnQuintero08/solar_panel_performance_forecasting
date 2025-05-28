import pandas as pd
import json

def load_json_data(location):
    with open(location, 'r') as data:
        data_json = json.load(data)
        
    new_data = data_json['properties']['parameter']
    df = pd.DataFrame(new_data).reset_index(names=['date'])
    df.columns = ['date', 'irradiation', 'temperature', 'wind_speed', 'precipitation']
    
    return df



location = 'data/initial_data.json'
df = load_json_data(location)