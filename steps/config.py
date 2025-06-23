#from zenml.steps import BaseParameters
from pydantic import BaseModel
class ModelNameConfig(BaseModel):
    """
    Model Congigs
    """
    model_name: str = "LinearRegression"