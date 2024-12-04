import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from azureml.core import Workspace
from datetime import datetime

class DataPreparation:
    """
    Class for preparing data. Includes loading, cleaning, and splitting.
    This is part of the Data Preparation stage of the ML lifecycle.
    """
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        """Load the dataset from the specified path..."""
        df = pd.read_csv(self.data_path)
        print(f"Data loaded successfully from {self.data_path}")
        return df

    def preprocess_data(self, df):
        """Preprocess the dataset by separating features and target."""
        # Separate features and target
        X = df.drop(columns=["Energy_Requirement"])
        y = df["Energy_Requirement"].apply(lambda x: 1 if x == "Yes" else 0)
        print("Data preprocessing completed.")
        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split the dataset into training and testing sets."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        print("Data split into training and testing sets.")
        return X_train, X_test, y_train, y_test

class ExperimentManager:
    """
    Class for managing ML experiments with multiple models.
    This is part of the Model Training and Experimentation stage of the ML lifecycle.
    """
    def __init__(self, experiment_name, workspace_config):
        self.experiment_name = experiment_name
        # Load the Azure ML workspace
        self.ws = Workspace.from_config(workspace_config)
        # Set up MLflow for tracking
        mlflow.set_tracking_uri(self.ws.get_mlflow_tracking_uri())
        mlflow.set_experiment(self.experiment_name)
        print(f"Experiment '{self.experiment_name}' is set up in MLflow.")

    def train_and_log_models(self, models, X_train, y_train, X_test, y_test):
        """
        Train and log multiple models in MLflow.
        Logs model parameters, metrics, and artifacts..
        """
        for model_name, model in models.items():
            # Dynamically name the run based on the model and current date
            run_name = f"{model_name}-{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
            with mlflow.start_run(run_name=run_name):
                print(f"Training and logging model: {model_name}")
                # Log model parameters
                mlflow.log_param("model_name", model_name)
                

                # Train the model
                model.fit(X_train, y_train)

                # Predict on test data
                y_pred = model.predict(X_test)

                # Evaluate the model
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred)
                }

                # Log metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

                # Log the trained model
                mlflow.sklearn.log_model(model, artifact_path="models")

                # Print logged metrics for debugging
                print(f"Model: {model_name}, Metrics: {metrics}")

        print(f"Experiment '{self.experiment_name}' completed. Check Azure ML Studio for results.")

class DataSaver:
    """
    Class for saving prediction data for future batch prediction or testing.
    This is part of the Deployment/Inference Preparation stage of the ML lifecycle.
    """
    @staticmethod
    def save_data(X_test, save_path):
        """Save the test dataset for predictions."""
        X_test.to_csv(save_path, index=False)
        print(f"Prediction data saved at: {save_path}")
