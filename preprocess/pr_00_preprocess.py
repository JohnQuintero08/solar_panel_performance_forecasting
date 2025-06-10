from preprocess.pr_01_dataset_construction import load_json_data
from preprocess.pr_02_format_data import format_columns
from preprocess.pr_03_clean_irregular_data import clean_irregular_data

location = 'data/initial_data.json'
df = load_json_data(location)
df_c = format_columns(df)
df_c = clean_irregular_data(df_c)
# TODO AÑADIR UNA FUNCION QUE ELIMINE VALORES NULOS O DUPLICADOS
df_c.to_feather('data/preprocess_data.feather')