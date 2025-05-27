import pandas as pd
import json

with open('data/initial_data.json', 'r') as data:
    data_json = json.load(data)
    
new_data = data_json['properties']['parameter']

df = pd.DataFrame(new_data).reset_index()