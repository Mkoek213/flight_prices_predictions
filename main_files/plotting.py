from main_files import plt
from main_files import skplt


def plot_residuals(model, X_train, X_test, y_train, y_test):

    # Predict target values using the trained model
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Create a new figure with specified size
    plt.figure(figsize=(10 ,10))

    # Create the first subplot for training data
    plt.subplot(1, 2, 1)

    # Scatter plot of true vs predicted values for training data
    plt.scatter(y_train, y_train_pred, c='crimson')

    # Set both x-axis and y-axis to logarithmic scale for better visualization
    plt.yscale('log')
    plt.xscale('log')

    # Define the range for the reference line
    p1 = max(max(y_train_pred), max(y_train))
    p2 = min(min(y_train_pred), min(y_train))

    # Plot a reference line where predictions would equal true values
    plt.plot([p1, p2], [p1, p2], 'b-')

    # Set title and labels for the first subplot
    plt.title('Training Plot')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')

    # Create the second subplot for test data
    plt.subplot(1, 2, 2)

    # Scatter plot of true vs predicted values for test data
    plt.scatter(y_test, y_test_pred, c='crimson')

    # Set both x-axis and y-axis to logarithmic scale for better visualization
    plt.yscale('log')
    plt.xscale('log')

    # Define the range for the reference line
    p3 = max(max(y_test_pred), max(y_test))
    p4 = min(min(y_test_pred), min(y_test))

    # Plot a reference line where predictions would equal true values
    plt.plot([p3, p4], [p3, p4], 'b-')

    # Set title and labels for the second subplot
    plt.title('Test Plot')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')

    # Adjust layout to prevent overlap of subplots
    plt.tight_layout()

    # Save the figure to a file
    plt.savefig('data_info_and_plots/plot_residuals.png')

    # Close the plot to free up memory
    plt.close()


def plot_learning_curve_for_model(model, X_train, y_train):

    # Generate the learning curve plot for the given model using scikitplot
    skplt.estimators.plot_learning_curve(model, X_train, y_train)

    # Set the title of the learning curve plot
    plt.title('Training Learning Curve')

    # Adjust layout to prevent overlap of subplots
    plt.tight_layout()

    # Save the figure to a file
    plt.savefig('data_info_and_plots/plot_learning_curve.png')

    # Close the plot to free up memory
    plt.close()


