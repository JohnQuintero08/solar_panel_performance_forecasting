import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

general_plot_size = (15,6)

def show_info(df):
    print("""
          ---GENERAL INFO---
          """)
    print(df.info())
    print("""
          ---HEAD---
          """)
    print(df.head())
    print("""
          ---DESCRIBE---
          """)
    print(df.describe())

def check_null_values(df):
    print('---NULL VALUES---')
    print(df.isna().sum())

def check_duplicates(df):
    print(f' Total number of rows duplicated {df.duplicated().sum()}')

def plot_pairplot(df, columns_to_drop = ['date']):
    sns.pairplot(df.drop(columns_to_drop, axis=1))
    plt.show()

def plot_variable_hist(df, param):
    plt.figure(figsize=general_plot_size)
    sns.histplot(data=df, x=param, bins=100, kde=True)
    plt.title(f'Histogram of {param.capitalize()}')
    plt.xlabel(param.capitalize())
    plt.ylabel('Frequency')
    plt.show()

def create_time_features(df):
    df_c = df.copy()
    df_c['year'] = df_c['date'].dt.year
    df_c['month'] = df_c['date'].dt.month
    df_c['day'] = df_c['date'].dt.day
    df_c['day_of_week'] = df_c['date'].dt.day_of_week
    return df_c

def plot_scatter_trim_by_year(df, param, year=1999):
    plt.figure(figsize=general_plot_size)
    plt.scatter(df.query(f'year > {year}')['date'], df.query(f'year > {year}')[param],
                s=2,)
    plt.title(f'{param.capitalize()} distribution over time')
    plt.ylabel(param.capitalize())
    plt.xlabel('Year')
    plt.show()

def plot_seasonal_param(df, param, year=1999):
    plt.figure(figsize=general_plot_size)
    sns.lineplot(data=df.query(f'year > {year}') ,
                 x='month', 
                 y=param, 
                 hue='year', 
                 palette='deep',
                 errorbar=None #Intervalo de confianza
                 )
    plt.show()


def plot_seasonal_param_polar_trim(df, param, year=1999):
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    years = df.query(f'year > {year}')['year'].unique()

    for i in sorted(years):
        sub = df[df['year'] == i]
        monthly_average = sub.groupby('month')[param].mean()
        theta = np.linspace(0, 2 * np.pi, len(monthly_average))
        radii = monthly_average.values
        
        ax.plot(theta, radii, label=str(i))

    ax.set_theta_direction(-1) 
    ax.set_theta_offset(np.pi / 2.0) 

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    ax.set_xticks(np.linspace(0, 2*np.pi, 12, endpoint=False))
    ax.set_xticklabels(months)

    ax.set_title(f'Estational graph of {param} in polar coordinates', va='bottom')
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    plt.tight_layout()
    plt.show()


def multi_lag_plot_trim(df, param, year=1999):    
    fig, axs = plt.subplots(3,4, figsize=(15,9))
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        pd.plotting.lag_plot(df.query(f'year > {year}')[param], lag=(i+1)*180,ax=ax)
        ax.set_title(f"Lag {(i + 1)*180}")
    plt.title(f'Lag plot of {param}')
    plt.tight_layout()
    plt.show()

def lag_plot_trim(df, param, lag):
    fig, ax = plt.subplots(figsize=(16,5))
    plot_acf(df[param], lags=lag,ax=ax)
    plt.show()

def plot_box(df, param):
    fig, axs = plt.subplots(1,2, figsize=general_plot_size)
    sns.boxplot(data = df.query('year <= 2012'), x='month', y=param, ax=axs[0])
    sns.boxplot(data = df.query('year > 2012'), x='month', y=param, ax=axs[1])
    plt.show()
    
    
# IMPORTANTEEEEE
# # - There is an irradiation data of -999, which is impossible, because the measure is greater than 0
# TODO CREAR UNA FUNCIÓN QUE ELIMINE LOS DATOS QUE TENGAN VELOCIDAD MENOR A CERO, IRRADIACION MENOR A CERO
df = pd.read_feather('data/preprocess_data.feather')
df= df.query('irradiation > 0')

plot_variable_hist(df, 'irradiation')
new_df = create_time_features(df)
plot_scatter_trim_by_year(new_df,'precipitation',2012)
plot_seasonal_param(new_df,'irradiation', 2012)
plot_seasonal_param_polar_trim(new_df, 'irradiation', 2012)
multi_lag_plot_trim(new_df, 'irradiation', 2012)
lag_plot_trim(new_df, 'irradiation', 365)
plot_box(new_df, 'irradiation')









