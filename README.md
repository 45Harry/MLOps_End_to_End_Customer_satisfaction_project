# MLops: Customer Satisfaction Prediction Pipeline

This repository implements a full MLOps workflow for predicting customer satisfaction using the Olist e-commerce dataset. It demonstrates best practices in data ingestion, cleaning, model development, evaluation, deployment, experiment tracking, and web-based inference, leveraging ZenML, MLflow, and Streamlit.

---

## Project Description

The goal of this project is to build a robust, modular, and production-ready machine learning pipeline for predicting customer satisfaction scores. The pipeline covers the entire ML lifecycle, from data ingestion and preprocessing to model training, evaluation, deployment, and serving predictions via a web interface. The project uses ZenML for pipeline orchestration, MLflow for experiment tracking, and Streamlit for interactive model inference.

---

## Requirements

- Python 3.8+
- Recommended: virtual environment (venv, conda, etc.)
- Install dependencies:
  ```bash
  pip install -r data/mlops-projects-course-main/customer-satisfaction-mlops-main/requirements.txt
  ```

---

## How to Use

1. **Clone the repository:**
   ```bash
   git clone https://github.com/45Harry/MLOps_End_to_End_Customer_satisfaction_project
   cd MLops
   ```

2. **Install dependencies:**
   ```bash
   pip install -r data/mlops-projects-course-main/customer-satisfaction-mlops-main/requirements.txt
   ```

3. **Run the training pipeline:**
   ```bash
   python run_pipeline.py
   ```

4. **Run the deployment pipeline:**
   ```bash
   python run_deployment.py
   ```

5. **Launch the Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```

6. **View MLflow UI:**
   ```bash
   mlflow ui --backend-store-uri "file:///tmp/mlflow"
   ```

---

## Repository Structure

### Root Directory

- **`run_pipeline.py`**  
  Entry point to run the training pipeline. Loads data, runs the training pipeline, and prints the MLflow tracking URI.

- **`run_deployment.py`**  
  Entry point to run the deployment pipeline. Supports deployment, prediction, or both, and manages the MLflow model server.

- **`streamlit_app.py`**  
  Streamlit web app for interactive model inference. Users can input product/order features and get satisfaction predictions.

- **`__init__.py`**  
  Marks the directory as a Python package.

- **`saved_model/`**  
  (Currently empty) Intended for storing serialized models.

- **`mlruns/`**  
  MLflow experiment tracking and model registry. Contains experiment metadata and run artifacts.

- **`pipelines/`**  
  Contains pipeline definitions and utilities:
  - `training_pipeline.py`: Defines the training pipeline (ingest, clean, train, evaluate).
  - `deployment_pipeline.py`: Defines the deployment and inference pipelines, including model serving logic.
  - `utils.py`: Utility functions for pipelines, e.g., test data preparation.

- **`steps/`**  
  Modular pipeline step scripts:
  - `ingest_data.py`: Loads data from a CSV file.
  - `clean_data.py`: Cleans and splits data into train/test sets.
  - `model_train.py`: Trains a model (currently supports Linear Regression).
  - `evaluation.py`: Evaluates the model (R2, RMSE, MSE).
  - `config.py`: Configuration class for model selection.

- **`src/`**  
  Source code for core logic:
  - `data_cleaning.py`: Data cleaning and preprocessing strategies.
  - `mode_dev.py`: Model abstraction and Linear Regression implementation.
  - `evaluation.py`: Evaluation metric classes (MSE, RMSE, R2).

- **`data/`**  
  Contains datasets and subprojects:
  - `archive/`: Full Olist dataset as CSV files.
  - `olist_customers_dataset.csv`: Standalone customer dataset.
  - `mlops-projects-course-main/`: Contains the main MLOps subproject (see below).

---

### Key Subdirectories

#### data/archive/

- Contains the full Olist dataset as CSV files:
  - `olist_customers_dataset.csv`
  - `olist_geolocation_dataset.csv`
  - `olist_order_items_dataset.csv`
  - `olist_order_payments_dataset.csv`
  - `olist_order_reviews_dataset.csv`
  - `olist_orders_dataset.csv`
  - `olist_products_dataset.csv`
  - `olist_sellers_dataset.csv`
  - `product_category_name_translation.csv`

#### data/mlops-projects-course-main/customer-satisfaction-mlops-main/

A self-contained MLOps project with its own pipeline, model, and app code:
- **`_assets/`**: Project diagrams and images.
- **`config.yaml`**: Pipeline configuration.
- **`data/`**: Project-specific data.
- **`materializer/`**: Custom ZenML materializer for data serialization.
- **`model/`**: Model development and evaluation scripts.
- **`pipelines/`**: Pipeline definitions.
- **`saved_model/`**: Serialized models.
- **`steps/`**: Modular pipeline steps.
- **`tests/`**: Unit tests.
- **`streamlit_app.py`**: Streamlit web app for model inference.
- **`run_pipeline.py`**: Script to run the training pipeline.
- **`run_deployment.py`**: Script to run the deployment pipeline.
- **`requirements.txt`**: Python dependencies.
- **`README.md`**: Subproject documentation.

---

## File and Folder Details

### Pipelines

- **`pipelines/training_pipeline.py`**  
  Defines the training pipeline: data ingestion, cleaning, model training, and evaluation.

- **`pipelines/deployment_pipeline.py`**  
  Defines the deployment and inference pipelines, including model serving, deployment triggers, and prediction logic.

- **`pipelines/utils.py`**  
  Utility functions for pipelines, such as preparing test data.

### Steps

- **`steps/ingest_data.py`**  
  Loads data from a specified CSV file path.

- **`steps/clean_data.py`**  
  Cleans the data and splits it into training and test sets using strategies from `src/data_cleaning.py`.

- **`steps/model_train.py`**  
  Trains a model (currently supports Linear Regression) and logs experiments to MLflow.

- **`steps/evaluation.py`**  
  Evaluates the trained model using R2, RMSE, and MSE metrics, and logs results to MLflow.

- **`steps/config.py`**  
  Contains the `ModelNameConfig` class for specifying the model type.

### Source Code

- **`src/data_cleaning.py`**  
  Implements data cleaning and preprocessing strategies, including handling missing values and splitting data.

- **`src/mode_dev.py`**  
  Abstract model class and Linear Regression implementation.

- **`src/evaluation.py`**  
  Abstract evaluation class and implementations for MSE, RMSE, and R2 metrics.

### MLflow

- **`mlruns/`**  
  MLflow experiment tracking directory. Contains experiment metadata and run artifacts.

---

## License

This project is licensed under the MIT License.

---

