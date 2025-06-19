from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from features.f_05_features_target_split import features_target_split

def metrics_eval(target, predictions, dataset_name=""):
    rmse = root_mean_squared_error(target, predictions)
    mae = mean_absolute_error(target, predictions)
    print(f'Dataset - {dataset_name}')
    print(f'RMSE: {rmse:.2f}')
    print(f'MAE: {mae:.2f}')
    return {
        f'rmse_{dataset_name}' : rmse,
        f'mae_{dataset_name}' : mae
    }


def graph_predictions(target, predictions, ax, title):
    ax.scatter(target, predictions, s=20, alpha=0.8)
    ax.plot([target.min(), target.max()], [target.min(), target.max()], 'r--')
    ax.set_title(title)
    ax.set_xlabel('Real value')
    ax.set_ylabel('Prediction')
    ax.legend(['Predictions', 'Ideal correlation'])


def model_evaluation_mlflow(model, df_train, df_pr, target_name, model_name,has_plot=False, df_name_1 ="train", df_name_2="valid"):

    f_train, t_train = features_target_split(df_train, target_name)
    f_pr, t_pr = features_target_split(df_pr, target_name)

    model.fit(f_train, t_train)
    
    predictions_t = model.predict(f_train)
    predictions_pr = model.predict(f_pr)

    print(f"Model evaluation: {model_name}")
    metrics_t = metrics_eval(t_train, predictions_t, df_name_1)
    metrics_pr = metrics_eval(t_pr, predictions_pr,df_name_2)
    metrics = {**metrics_t, **metrics_pr}
    
    if has_plot:
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        graph_predictions(t_train, predictions_t, axs[0], df_name_1)
        graph_predictions(t_pr, predictions_pr, axs[1], df_name_2)

        fig.suptitle(f'Comparison of the predictions - {model_name}', fontsize=16)
        plt.tight_layout()
        plt.show()
        
    return metrics


def arima_model_evaluation_mlflow(model, df_to_predict, step, model_name, has_plot):
    results = model.fit()
    predictions = results.forecast(steps= step)
    print(f"Model evaluation: {model_name}")
    metrics = metrics_eval(df_to_predict, predictions, 'valid')
    
    if has_plot:
        plt.scatter(df_to_predict, predictions, s=20, alpha=0.8)
        plt.plot([df_to_predict.min(), df_to_predict.max()], [df_to_predict.min(), df_to_predict.max()], 'r--')
        plt.title("SARIMA predictions for irradiation")
        plt.xlabel('Real value')
        plt.ylabel('Prediction')
        plt.legend(['Predictions', 'Ideal correlation'])
        plt.tight_layout()
        plt.show()
    return metrics
