import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from unittest.mock import MagicMock, patch

import sys
sys.path.append('../src/')

from src.train import DataPreparation, ExperimentManager, DataSaver  # Replace `your_module` with the actual file name

@pytest.fixture
def sample_data():
    """Fixture to provide sample data for testing."""
    data = {
        "Temperature (°C)": [25.4, 18.6, 30.1],
        "Humidity (%)": [45, 60, 55],
        "Energy Usage (kWh)": [15.2, 20.5, 10.1],
        "Peak Hours (hours)": [2.5, 1.8, 3.0],
        "Cost of Energy ($)": [0.12, 0.15, 0.10],
        "Weather Index": [1.8, 2.1, 1.5],
        "Appliance Load (%)": [70, 85, 60],
        "Solar Output (kWh)": [3.2, 1.8, 4.5],
        "Wind Output (kWh)": [2.4, 1.2, 3.0],
        "Seasonality Index": [1.3, 1.5, 1.2],
        "Energy_Requirement": ["Yes", "No", "Yes"]
    }
    return pd.DataFrame(data)

def test_load_data(sample_data, tmp_path):
    """Test the load_data method in DataPreparation."""
    # Save mock data to a temporary CSV
    test_file = tmp_path / "test_data.csv"
    sample_data.to_csv(test_file, index=False)

    # Load data using DataPreparation
    dp = DataPreparation(data_path=str(test_file))
    df = dp.load_data()

    # Validate loaded data
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3  # Ensure 3 rows are loaded

def test_preprocess_data(sample_data):
    """Test the preprocess_data method in DataPreparation."""
    dp = DataPreparation(data_path=None)  # No path needed for this test
    df = sample_data

    # Preprocess data
    X, y = dp.preprocess_data(df)

    # Validate features and target
    assert isinstance(X, pd.DataFrame)
    assert len(X.columns) == 10  # 10 features
    assert all(y == [1, 0, 1])  # Ensure target is correctly encoded

def test_split_data(sample_data):
    """Test the split_data method in DataPreparation."""
    dp = DataPreparation(data_path=None)
    df = sample_data
    X, y = dp.preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = dp.split_data(X, y)

    # Validate split
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

@patch("mlflow.start_run")
@patch("mlflow.set_experiment")
@patch("mlflow.log_param")
@patch("mlflow.log_metric")
@patch("mlflow.sklearn.log_model")
def test_train_and_log_models(
    mock_log_model, mock_log_metric, mock_log_param, mock_set_experiment, mock_start_run, sample_data
):
    """Test the train_and_log_models method in ExperimentManager."""
    
    if not os.path.exists("./config.json"):
        pytest.skip("Skipping test because config.json is missing")

    # Test logic here
    pass
    mock_workspace = MagicMock()
    mock_workspace.get_mlflow_tracking_uri.return_value = "mock_uri"

    # Create ExperimentManager
    experiment_manager = ExperimentManager(
        experiment_name="test_experiment",
        workspace_config="mock_config"
    )
    experiment_manager.ws = mock_workspace

    # Prepare data
    dp = DataPreparation(data_path=None)
    df = sample_data
    X, y = dp.preprocess_data(df)
    X_train, X_test, y_train, y_test = dp.split_data(X, y)

    # Define models
    models = {
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Support Vector Machine": SVC()
    }

    # Train and log models
    experiment_manager.train_and_log_models(models, X_train, y_train, X_test, y_test)

    # Validate that MLflow methods were called
    assert mock_start_run.called
    assert mock_log_param.called
    assert mock_log_metric.called
    assert mock_log_model.called

def test_save_data(tmp_path, sample_data):
    """Test the save_data method in DataSaver."""
    # Save the test dataset
    test_file = tmp_path / "test_predictions.csv"
    X = sample_data.drop(columns=["Energy_Requirement"])

    # Save data
    DataSaver.save_data(X, str(test_file))

    # Validate saved data
    saved_df = pd.read_csv(test_file)
    assert isinstance(saved_df, pd.DataFrame)
    assert len(saved_df) == len(X)
    assert list(saved_df.columns) == list(X.columns)
