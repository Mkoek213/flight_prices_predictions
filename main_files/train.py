from main_files import GridSearchCV
from main_files import LinearRegression
from main_files import mean_squared_error
from main_files import r2_score
from main_files import train_test_split

def train_linear_regression(X_train, X_test, y_train, y_test):
    parameters = {
        'fit_intercept': [True, False]
    }

    grid_search = GridSearchCV(
        estimator=LinearRegression(),
        param_grid=parameters,
        scoring='r2',
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    model = LinearRegression(**best_params)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"Best score: {best_score}")
    print(f"Train MSE: {train_mse}")
    print(f"Test MSE: {test_mse}")
    print(f"Train R^2: {train_r2}")
    print(f"Test R^2: {test_r2}")

    return model

