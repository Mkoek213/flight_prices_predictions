from main_files import DecisionTreeRegressor
from main_files import XGBRegressor
from main_files import np
from main_files import GridSearchCV
from main_files import LinearRegression
from main_files import mean_squared_error
from main_files import r2_score
from main_files import sp_randInt
from main_files import sp_randFloat
from main_files import RandomizedSearchCV
from main_files import time
from main_files import joblib

def train_linear_regression(X_train, X_test, y_train, y_test):
    # Define parameter grid for GridSearchCV
    parameters = {
        'fit_intercept': [True, False]
    }

    # Initialize GridSearchCV with LinearRegression and the parameter grid
    grid_search = GridSearchCV(
        estimator=LinearRegression(),
        param_grid=parameters,
        scoring='r2',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    # Start timing for model training
    start_time = time.time()

    # Perform Grid Search to find the best parameters
    grid_search.fit(X_train, y_train)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Retrieve the best parameters and score from Grid Search
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Train the LinearRegression model with the best parameters
    model = LinearRegression(**best_params)
    model.fit(X_train, y_train)

    # Predict on training and testing data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate performance metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Print model performance and training time
    print(f"Best score: {best_score}")
    print(f"Training time: {elapsed_time:.2f} seconds")
    print(f"Train MSE: {train_mse}")
    print(f"Test MSE: {test_mse}")
    print(f"Train R^2: {train_r2}")
    print(f"Test R^2: {test_r2}")

    # Save the trained model to a file
    model_path = 'models/linear_regression_model.joblib'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def train_xgbregressor(X_train, X_test, y_train, y_test):
    # Define parameter distributions for RandomizedSearchCV
    parameters = {
        'learning_rate': np.linspace(0.0001, 0.2, 100),
        'subsample': sp_randFloat(0.8, 0.2),
        'n_estimators': np.arange(100, 2001, 50),
        'max_depth': [2, 3, 4, 5, 6, 7],
        'min_child_weight': [1,2,3,4]
    }

    # Initialize RandomizedSearchCV with XGBRegressor and the parameter distributions
    random_search = RandomizedSearchCV(
        estimator=XGBRegressor(),
        n_iter = 10,
        param_distributions = parameters,
        scoring='r2',
        cv=5,
        n_jobs=-1,
        verbose=4,
        random_state = 1
    )

    # Start timing for model training
    start_time = time.time()

    # Perform Randomized Search to find the best parameters
    random_search.fit(X_train, y_train)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Retrieve the best parameters and score from Randomized Search
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    # Train the XGBRegressor model with the best parameters
    model = XGBRegressor(**best_params)
    model.fit(X_train, y_train)

    # Predict on training and testing data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate performance metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Print model performance and training time
    print(f"Best score: {best_score}")
    print(f"Training time: {elapsed_time:.2f} seconds")
    print(f"Train MSE: {train_mse}")
    print(f"Test MSE: {test_mse}")
    print(f"Train R^2: {train_r2}")
    print(f"Test R^2: {test_r2}")

    # Save the trained model to a file
    model_path = 'models/xgbregressor_model.joblib'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def train_tree_regressor(X_train, X_test, y_train, y_test):
    # Define parameter distributions for RandomizedSearchCV
    parameters = {
        'splitter': ['random'],
        'max_depth': sp_randInt(10, 25),
        'min_samples_split': sp_randInt(15, 50),
        'max_features': np.linspace(800, 1500, 25).astype(int),
        'criterion': ['friedman_mse'],
        'min_samples_leaf': sp_randInt(1, 5),
        'min_weight_fraction_leaf': sp_randFloat(0, 0.001),
        'max_leaf_nodes': sp_randInt(100, 250),
        'min_impurity_decrease': sp_randFloat(0.1, 0.6)
    }

    # Initialize RandomizedSearchCV with DecisionTreeRegressor and the parameter distributions
    random_search = RandomizedSearchCV(
        estimator=DecisionTreeRegressor(),
        n_iter = 100,
        param_distributions = parameters,
        scoring='r2',
        cv=5,
        n_jobs=-1,
        verbose=4,
        random_state = 1
    )

    # Start timing for model training
    start_time = time.time()

    # Perform Randomized Search to find the best parameters
    random_search.fit(X_train, y_train)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Retrieve the best parameters and score from Randomized Search
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    # Train the DecisionTreeRegressor model with the best parameters
    model = DecisionTreeRegressor(**best_params)
    model.fit(X_train, y_train)

    # Predict on training and testing data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate performance metrics

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Print model performance and training time
    print(f"Best score: {best_score}")
    print(f"Training time: {elapsed_time:.2f} seconds")
    print(f"Train MSE: {train_mse}")
    print(f"Test MSE: {test_mse}")
    print(f"Train R^2: {train_r2}")
    print(f"Test R^2: {test_r2}")

    # Save the trained model to a file
    model_path = 'models/tree_regressor_model.joblib'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
