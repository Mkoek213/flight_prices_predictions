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
from sklearn.model_selection import train_test_split # For splitting the dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
import time
import joblib
from main_files.data_processing import one_hot_encoding_category_data, process_time_data, print_data_info, \
    split_scale_data
from main_files.train import train_linear_regression, train_xgbregressor

if __name__ == "__main__":
    flight_data = pd.read_csv('flight_dataset.csv')
    flight_data_processed = one_hot_encoding_category_data(flight_data)
    # print_data_info(processed_data)
    # plot_for_outliers(flight_data)
    # plot_correlation_heatmap(flight_data)
    # plot_distribution_info(flight_data)
    # plot_category_data_distribution(flight_data)
    flight_data_processed = process_time_data(flight_data_processed)
    print_data_info(flight_data_processed)
    X_train, X_test, y_train, y_test = split_scale_data(flight_data_processed, 'Price')

    # train_linear_regression(X_train, X_test, y_train, y_test)
    # train_xgbregressor(X_train, X_test, y_train, y_test)

    linear_regression_model = joblib.load('models/linear_regression_model.joblib')
    xgbregression_model = joblib.load('models/xgbregressor_model.joblib')
