import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from matplotlib import pyplot as plt

def plot_decompose(decomposition, method_name, param,  save_fig=False):
    fig, axs = plt.subplots(4, 1, figsize=(15, 10), sharex=True)

    axs[0].plot(decomposition.observed)
    axs[0].set_title('Observed')

    axs[1].plot(decomposition.trend)
    axs[1].set_title('Trend')

    axs[2].plot(decomposition.seasonal)
    axs[2].set_title('Seasonal')

    axs[3].plot(decomposition.resid)
    axs[3].set_title('Residual')

    plt.suptitle(f'Time series decomposition using {method_name}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    if save_fig:
        plt.savefig(f'data/plots/seasonal_decomposition_{param}.png')
    plt.show()

def print_residual(decompose):
    print(f'Residual: {decompose.resid.mean():.7f}')
    
    
def classic_decompose(df, param, period, model_type='additive'):
    decompose = seasonal_decompose(df[param], model=model_type, period=period)
    return decompose
    
    
def stl_decompose(df, param ,period):
    decompose = STL(df[param], period=period).fit()
    return decompose
    
    
def check_residuals_by_period(df, param, method):
    results = []
    list_range = [2,10,50,100,180,360]
    for per in list_range:
        decompose = method(df, param, per)
        results.append((per, decompose.resid.mean()))
    df_r = pd.DataFrame(np.abs(results), columns=['period', 'residual_mean'])
    df_r.sort_values('residual_mean')
    return df_r

def setup_df_decompose(df):
    df_c = df.copy()
    df_c = df_c.set_index('date', drop=True)
    df_c = df_c.asfreq('D', method='ffill')
    return df_c
    