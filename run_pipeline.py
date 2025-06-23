from readline import backend

from pipelines.training_pipeline import training_pipeline
from zenml.client import Client


if __name__ =='__main__':
    #Run the pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipeline(data_path='/home/harry/Documents/Code/Data_Science/MLops/data/olist_customers_dataset.csv')


mlflow ui --backend-store-uri "file:///tmp/mlflow"