import tempfile
from unittest.mock import patch

import joblib
import pytest
import pandas as pd
import os
import sys
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

sys.path.insert(0, 'main_files')

from main_files.data_processing import one_hot_encoding_category_data, process_time_data, split_scale_data
from main_files.plotting import plot_residuals, plot_learning_curve_for_model, plot_feature_importance_for_model, plot_partial_dependence

from main_files.train import train_linear_regression, train_xgbregressor, train_tree_regressor


@pytest.fixture
def setup_data():
    flight_data = pd.read_csv('main_files/flight_dataset.csv')
    flight_data_processed = one_hot_encoding_category_data(flight_data)
    flight_data_processed2 = process_time_data(flight_data_processed)
    X_train, X_test, y_train, y_test = split_scale_data(flight_data_processed, 'Price')

    return flight_data, flight_data_processed, flight_data_processed2, X_train, X_test, y_train, y_test


def test_one_hot_encoding_category_data(setup_data):
    flight_data, flight_data_processed, flight_data_processed2, X_train, X_test, y_train, y_test= setup_data
    # Test one_hot_encoding_category_data function
    encoded_data = one_hot_encoding_category_data(flight_data)
    assert isinstance(encoded_data, pd.DataFrame)


def test_process_time_data(setup_data):
    flight_data, flight_data_processed, flight_data_processed2, X_train, X_test, y_train, y_test = setup_data
    # Test process_time_data function
    processed_data = process_time_data(flight_data_processed)
    assert isinstance(processed_data, pd.DataFrame)


def test_split_scale_data(setup_data):
    flight_data, flight_data_processed, flight_data_processed2, X_train, X_test, y_train, y_test = setup_data
    # Test split_scale_data function
    X_train_scaled, X_test_scaled, y_train, y_test = split_scale_data(flight_data_processed2, 'Price')
    assert X_train_scaled.shape[0] == X_train.shape[0]
    assert X_test_scaled.shape[0] == X_test.shape[0]

def test_train_linear_regression(setup_data):
    flight_data, flight_data_processed, flight_data_processed2, X_train, X_test, y_train, y_test = setup_data
    # Create a temporary directory
    # Run the function that saves the model
    model = train_linear_regression(X_train, X_test, y_train, y_test)

    assert model is not None
    assert isinstance(model, LinearRegression)  # Check if the returned object is a LinearRegression model


def test_train_xgbregressor(setup_data):
    flight_data, flight_data_processed, flight_data_processed2, X_train, X_test, y_train, y_test = setup_data
    # Create a temporary directory
    # Run the function that saves the model
    model = train_xgbregressor(X_train, X_test, y_train, y_test)

    assert model is not None
    assert isinstance(model, XGBRegressor)  # Check if the returned object is a LinearRegression model



def test_train_tree_regressor(setup_data):
    flight_data, flight_data_processed, flight_data_processed2, X_train, X_test, y_train, y_test = setup_data
    # Create a temporary directory
    # Run the function that saves the model
    model = train_tree_regressor(X_train, X_test, y_train, y_test)

    assert model is not None
    assert isinstance(model, DecisionTreeRegressor)  # Check if the returned object is a LinearRegression model



def test_plot_residuals(setup_data):
    flight_data, flight_data_processed, flight_data_processed2, X_train, X_test, y_train, y_test = setup_data
    # Test if plot_residuals function runs without errors
    model = train_linear_regression(X_train, X_test, y_train, y_test)
    try:
        plot_residuals(model, X_train, X_test, y_train, y_test)
    except Exception as e:
        pytest.fail(f'plot_residuals raised an exception: {e}')


def test_plot_learning_curve_for_model(setup_data):
    flight_data, flight_data_processed, flight_data_processed2, X_train, X_test, y_train, y_test = setup_data
    # Test if plot_residuals function runs without errors
    model = train_linear_regression(X_train, X_test, y_train, y_test)
    try:
        plot_learning_curve_for_model(model, X_train, y_train)
    except Exception as e:
        pytest.fail(f'plot_learning_curve_for_model raised an exception: {e}')


def test_plot_partial_dependence(setup_data):
    flight_data, flight_data_processed, flight_data_processed2, X_train, X_test, y_train, y_test = setup_data
    # Test if plot_residuals function runs without errors
    model = train_linear_regression(X_train, X_test, y_train, y_test)
    try:
        plot_partial_dependence(model, X_train)
    except Exception as e:
        pytest.fail(f'plot_partial_dependence raised an exception: {e}')

