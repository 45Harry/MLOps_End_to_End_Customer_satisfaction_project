import logging
import pandas as pd
from zenml import step
import mlflow

from src.mode_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker = experiment_tracker.name)
def train_model(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
        config: ModelNameConfig) -> RegressorMixin:
    """
    Train the model on the ingested data.
    Args:
         df: the ingested data

    """
    try:
        model = None
        if config.model_name == 'LinearRegression':
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train,y_train)
            return trained_model
        else:
            raise ValueError("Model {} not supported".format(config.model_name))
    except Exception as e:
        logging.error('Error in training model: {}'.format(e))
        raise e