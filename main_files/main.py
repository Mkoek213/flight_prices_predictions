from main_files import pd, one_hot_encoding_category_data, process_time_data, split_scale_data, print_data_info, plot_for_outliers, plot_correlation_heatmap, plot_distribution_info, \
    plot_category_data_distribution, train_xgbregressor, train_tree_regressor, train_linear_regression, joblib, plot_residuals, plot_partial_dependence, plot_feature_importance_for_model,\
    plot_learning_curve_for_model
import os

if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
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
    train_linear_regression(X_train, X_test, y_train, y_test)
    train_xgbregressor(X_train, X_test, y_train, y_test)
    train_tree_regressor(X_train, X_test, y_train, y_test)

    # Load pre-trained Linear Regression model from a file
    linear_regression_model = joblib.load('models/linear_regression_model.joblib')

    # Load pre-trained XGBRegressor model from a file
    xgbregression_model = joblib.load('models/xgbregressor_model.joblib')

    # Load pre-trained XGBRegressor model from a file
    tree_regressor_model = joblib.load('models/tree_regressor_model.joblib')

    # Uncomment these lines to plot model performance metrics
    plot_residuals(tree_regressor_model, X_train, X_test, y_train, y_test)
    plot_learning_curve_for_model(xgbregression_model, X_train, y_train)
    plot_feature_importance_for_model(xgbregression_model, X_train)
    plot_partial_dependence(xgbregression_model, X_train)