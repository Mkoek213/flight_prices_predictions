# Import necessary libraries and modules
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations and array manipulation
import warnings  # For controlling the visibility of warnings
import matplotlib.pyplot as plt  # For creating visualizations
import seaborn as sns  # For creating statistical data visualizations
from sklearn.model_selection import train_test_split  # For splitting data into training and test sets
from feature_engine.discretisation import EqualWidthDiscretiser  # For binning continuous data into discrete intervals
from feature_engine.encoding import OneHotEncoder  # For encoding categorical variables
from sklearn.preprocessing import StandardScaler  # For standardizing features by removing the mean and scaling to unit variance
from sklearn.linear_model import LinearRegression  # For implementing linear regression models
from sklearn.metrics import mean_squared_error  # For calculating the mean squared error of the model's predictions
from sklearn.metrics import r2_score  # For calculating the R^2 score of the model's predictions
from sklearn.ensemble import RandomForestRegressor  # For implementing a random forest regressor model
from sklearn.model_selection import GridSearchCV  # For performing grid search to find the best model parameters
from scipy.stats import uniform as sp_randFloat  # For generating random floats
from scipy.stats import randint as sp_randInt  # For generating random integers
from sklearn.model_selection import RandomizedSearchCV  # For performing randomized search to find the best model parameters
from xgboost import XGBRegressor  # For implementing XGBoost regression model
import time  # For measuring time
import joblib  # For saving and loading models
from main_files.data_processing import one_hot_encoding_category_data, process_time_data, print_data_info, \
    split_scale_data, plot_for_outliers, plot_correlation_heatmap, plot_distribution_info, \
    plot_category_data_distribution
from main_files.train import train_linear_regression, train_xgbregressor

if __name__ == "__main__":
    # Load the flight dataset from a CSV file
    flight_data = pd.read_csv('flight_dataset.csv')

    # Apply one-hot encoding to categorical variables in the dataset
    flight_data_processed = one_hot_encoding_category_data(flight_data)

    # Uncomment these lines to perform data visualization and analysis
    print_data_info(flight_data_processed)
    plot_for_outliers(flight_data)
    plot_correlation_heatmap(flight_data)
    plot_distribution_info(flight_data)
    plot_category_data_distribution(flight_data)

    # Process time-related features in the dataset
    flight_data_processed = process_time_data(flight_data_processed)

    # Print data information (e.g., statistics, data types)
    print_data_info(flight_data_processed)

    # Split data into training and test sets and scale features
    X_train, X_test, y_train, y_test = split_scale_data(flight_data_processed, 'Price')

    # Uncomment these lines to train the models
    # train_linear_regression(X_train, X_test, y_train, y_test)
    # train_xgbregressor(X_train, X_test, y_train, y_test)

    # Load pre-trained Linear Regression model from a file
    linear_regression_model = joblib.load('models/linear_regression_model.joblib')

    # Load pre-trained XGBRegressor model from a file
    xgbregression_model = joblib.load('models/xgbregressor_model.joblib')
