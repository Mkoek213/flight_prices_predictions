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

flight_data = pd.read_csv('flight_dataset.csv')
print(flight_data.info())
print(flight_data.head())