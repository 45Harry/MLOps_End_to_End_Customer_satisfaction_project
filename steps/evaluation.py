import logging
import pandas as pd
from zenml import step
from typing import Tuple

from src.evaluation import MSE,RMSE,R2Score
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from zenml.client import Client
import mlflow

#from steps.model_train import experiment_tracker

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluation_model(model: RegressorMixin,
                     X_test:pd.DataFrame,
                     y_test:pd.DataFrame) ->Tuple[
    Annotated[float,'r2score'],
    Annotated[float,'rmse'],
]:
    """
    Evaluate the model on the ingested data.
    Args:
        df: the ingested data

    """
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_score(y_test,prediction)
        mlflow.log_metrics({'mse':mse})

        r2_class = R2Score()
        r2 = r2_class.calculate_score(y_test,prediction)
        mlflow.log_metrics({'r2':r2})

        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test,prediction)
        mlflow.log_metrics({'rmse':rmse})
        return r2,rmse
    except Exception as e:
        logging.error(f"Error in evaluating model:{e}")
        raise e