from pickle import FALSE
import numpy as np
import pandas as pd
import logging
import json
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from pydantic import BaseModel
from pipelines.utils import get_data_for_test

from steps.clean_data import clean_data
from steps.evaluation import evaluation_model
from steps.ingest_data import ingest_data
from steps.model_train import train_model

docker_settings = DockerSettings(required_integrations=[MLFLOW])


class DeploymentTriggerConfig(BaseModel):
    """Deployment trigger config"""
    min_accuracy: float = 0.0

@step(enable_cache=False)
def dyanmic_importer()->str:
    data = get_data_for_test()
    return data




@step
def deployment_trigger(
        accuracy: float,
        config: DeploymentTriggerConfig,
) -> bool:
    """Implements a simple model deployment trigger that looks at the input model accuracy
    and decides if it is good enough to deploy or not"""
    return accuracy >= config.min_accuracy



class MLFlowDeploymentLoaderStepParameters(BaseModel):
    """mlflow deployment getter parameters"""
    pipeline_name:str
    step_name:str
    running:bool=True



@step(enable_cache=False)
def prediction_service_loader(
        pipeline_name: str,
        pipeline_step_name:str,
        running:bool = True,
        model_name: str = 'model'
) ->MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline.
    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction server
        step_name: the naem fo the step that deployed the MLflow prediction server
        running : when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
        """

    # get he mlflow dwployer stack component
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    # fetch existing services with same pipeline name, step name and model name

    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name = pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name = model_name,
        running = running
    )
    if not existing_services:
        raise RuntimeError(
            f"No MLflow deployment services found for pipeline {pipeline_name}, "
            f"step {pipeline_step_name} and model {model_name}."
            f"pipeline for the '{model_name}' model is currently "
            f"running"

        )
    return existing_services[0]


@step
def predictor(
        service: MLFlowDeploymentService,
        data: str,
) -> np.ndarray:
    try:
        # Start service if not running
        if not service.is_running:
            service.start(timeout=30)

        # Parse JSON data
        data_dict = json.loads(data)

        # Create DataFrame with correct columns
        df = pd.DataFrame(data_dict['data'], columns=data_dict['columns'])

        # Convert all numeric columns
        numeric_cols = [
            'payment_sequential', 'payment_installments', 'payment_value',
            'price', 'freight_value', 'product_name_lenght',
            'product_description_lenght', 'product_photos_qty',
            'product_weight_g', 'product_length_cm',
            'product_height_cm', 'product_width_cm'
        ]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        # Debug output
        print("Input data sample:")
        print(df.head())
        print("Data types:")
        print(df.dtypes)

        # Convert to the format MLflow expects
        # Option 1: Send DataFrame directly (recommended)
        prediction = service.predict(df)

        # Option 2: If that doesn't work, use this alternative format
        # input_data = df.values.tolist()  # Convert to list of lists
        # prediction = service.predict(input_data)

        return prediction

    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise







@pipeline(enable_cache=False, settings={'docker': docker_settings})
def continuous_deployment_pipeline(
        data_path: str,
        min_accuracy: float = 0.0,
        workers: int = 1,
        timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    # Data processing steps
    df = ingest_data(data_path=data_path)
    X_train, X_test, y_train, y_test = clean_data(df)

    # Model training and evaluation
    model = train_model(X_train, X_test, y_train, y_test, config={'learning_rate': 0.01, 'epochs': 10})
    r2_score, rmse = evaluation_model(model, X_test, y_test)

    # Deployment decision
    deployment_config = DeploymentTriggerConfig(min_accuracy=min_accuracy)
    deployment_decision = deployment_trigger(accuracy=r2_score, config=deployment_config)

    # Model deployment
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,  # Fixed parameter name
        workers=workers,
        timeout=timeout,
    )


@pipeline(enable_cache = False, settings={'docker':docker_settings})
def inference_pipeline(pipeline_name:str ,pipeline_step_name:str):
    data = dyanmic_importer()
    service = prediction_service_loader(
        pipeline_name = pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running = False
    )

    prediction = predictor(service=service,data = data)
    return prediction