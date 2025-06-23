import logging
from abc import ABC, abstractmethod
import numpy as np
import math

from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluattion our models
    """
    @abstractmethod
    def calculate_score(self,y_true:np.ndarray,y_pred:np.ndarray):
        """
        Calculates the scores for the model
        Args:
             y_true : True labels
             y_pred : Predicted labels
        Returns:
            None
        """
        pass

class MSE(Evaluation):
    # Evaluation strategy that uses mean squared error

    def calculate_score(self,y_true:np.ndarray,y_pred:np.ndarray):
        try:
            logging.info('Calculating MSE')
            mse = mean_squared_error(y_true,y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in calculating MSE: {e}")
            return e
class R2Score(Evaluation):
    """
    Evaluation Strategy that uses R2 Score
    """
    def calculate_score(self,y_true:np.ndarray,y_pred:np.ndarray):
        try:
            logging.info("Calculating R2 Score")
            r2 = r2_score(y_true,y_pred)
            logging.info(f"R2 Scores: {r2}")
            return r2
        except Exception as e:
            logging.error(f'Error in calculating R2 Score: {e}')
            return e

class RMSE(Evaluation):
    """
        Evaluation Strategy that uses Root Mean Squared Error
        """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE")
            mse = mean_squared_error(y_true, y_pred)
            rmse = math.sqrt(mse)
            logging.info(f"RMSE Scores: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f'Error in calculating RMSE: {e}')
            return e
