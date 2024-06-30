from main_files import flight_data
from main_files import plt
from main_files import sns
from main_files import np
from main_files import pd
import io
import sys


def print_data_info(database):
    buffer = io.StringIO()
    sys.stdout = buffer
    database.info()
    sys.stdout = sys.__stdout__
    info_str = buffer.getvalue()
    with open('data_info.txt', 'w') as f:
        f.write(f"Parameters info:\n{info_str}\n")
        f.write(f"Head rows:\n{database.head()}\n\n")
        f.write(f"Data description:\n{database.describe(include='all')}\n\n")
        f.write(f"Check for null values:\n{database.isnull().sum()}\n\n")

def plot_distribution_info(database):
    flight_data_plot_copy = database.copy()
    # plot only columns which dtype is not 'Object'
    columns_to_plot = [column for column in flight_data_plot_copy.columns if
                       flight_data_plot_copy[column].dtypes != 'O' and column != 'Year']

    figure, ax = plt.subplots(6, 2, figsize=(12, 9))
    ax = ax.flatten()

    for i, column in enumerate(columns_to_plot):
        if column == 'Price':
            sns.histplot(flight_data_plot_copy[column], kde=True, stat='percent', ax=ax[i], log_scale=True)
            custom_ticks = np.arange(2000, 26000, 6000)
            ax[i].set_xticks(custom_ticks)
            ax[i].set_xticklabels([f'{int(label)}' for label in custom_ticks])
        elif column == 'Total_Stops':
            bins = np.arange(0, 5, 1)
            sns.histplot(flight_data_plot_copy[column], kde=False, stat='percent', ax=ax[i], bins=bins)
            ax[i].set_xticks(bins)
            ax[i].set_xticklabels([f'{int(bin)}' for bin in bins])
        elif column == 'Month':
            bins = np.arange(0, 13, 1)
            sns.histplot(flight_data_plot_copy[column], kde=False, stat='percent', ax=ax[i], bins=bins)
            ax[i].set_xticks(bins)
            ax[i].set_xticklabels([f'{int(bin)}' for bin in bins])
        else:
            sns.histplot(flight_data_plot_copy[column], kde=True, stat='percent', ax=ax[i])

    for i in range(len(columns_to_plot), 12):
        ax[i].axis('off')

    plt.tight_layout()
    plt.savefig('plot_distribution.png')
    plt.close()

def plot_for_outliers(database):
    flight_data_plot_copy = database.copy()
    columns_to_plot = [column for column in flight_data_plot_copy.columns if
                       flight_data_plot_copy[column].dtypes != 'O' and column != 'Year']
    figure, ax = plt.subplots(3, 4, figsize=(12, 9))
    ax = ax.flatten()

    for i, column in enumerate(columns_to_plot):
        flight_data_plot_copy.boxplot(column, ax=ax[i])

    for i in range(len(columns_to_plot), 12):
        ax[i].axis('off')

    plt.tight_layout()
    plt.savefig('plot_outliers.png')
    plt.close()

def plot_correlation_heatmap(database):
    flight_data_plot_copy = database.copy()
    columns_to_plot = [column for column in flight_data_plot_copy.columns if
                       flight_data_plot_copy[column].dtypes != 'O' and column != 'Year']
    colormap = plt.cm.viridis
    plt.figure(figsize=(12, 12))
    plt.title('Correlation of Features', size=12)
    ax = sns.heatmap(flight_data_plot_copy[columns_to_plot].corr(), cmap=colormap, annot=True, vmin=-1, vmax=1)
    plt.savefig('plot_correlation_heatmap.png')
    plt.close()


def plot_category_data_distribution(database):
    flight_data_plot_copy = database.copy()
    category_columns = [column for column in flight_data_plot_copy.columns if
                        flight_data_plot_copy[column].dtypes == 'O']
    figure, ax = plt.subplots(3, 1, figsize=(12, 12))
    ax = ax.flatten()

    for i, col in enumerate(category_columns):
        sns.countplot(x=col, data=flight_data_plot_copy, ax=ax[i])
        ax[i].set_ylabel('Count')
        ax[i].set_title(f'Distribution of {col}', fontsize=14, weight='bold')
        ax[i].tick_params(axis='x', rotation=45, labelsize=10)
        ax[i].set_xlabel(None)

    plt.tight_layout()
    plt.savefig('plot_category_data_distribution.png')
    plt.close()

def one_hot_encoding_category_data(database):
    flight_data_processed = database.drop(columns=['Year'])

    # One hot encoding 'Airline' parameter
    value_counts_airline = flight_data_processed['Airline'].value_counts()
    top_airlines = value_counts_airline.head(8).index.tolist()
    flight_data_processed['Airline'] = flight_data_processed['Airline'].apply(
        lambda x: x if x in top_airlines else 'Others')
    df_encoded_airline = pd.get_dummies(flight_data_processed['Airline'], prefix='Airline').astype(int)
    flight_data_processed = pd.concat([flight_data_processed, df_encoded_airline], axis=1)
    flight_data_processed.drop('Airline', axis=1, inplace=True)

    # One hot encoding 'Source' parameter
    value_counts_source = flight_data_processed['Source'].value_counts()
    df_encoded_source = pd.get_dummies(flight_data_processed['Source'], prefix='Source').astype(int)
    flight_data_processed = pd.concat([flight_data_processed, df_encoded_source], axis=1)
    flight_data_processed.drop('Source', axis=1, inplace=True)

    # One hot encoding 'Destination' parameter
    value_counts_destination = flight_data_processed['Destination'].value_counts()
    df_encoded_destination = pd.get_dummies(flight_data_processed['Destination'], prefix='Destination').astype(int)
    flight_data_processed = pd.concat([flight_data_processed, df_encoded_destination], axis=1)
    flight_data_processed.drop('Destination', axis=1, inplace=True)

    return flight_data_processed


processed_data = one_hot_encoding_category_data(flight_data)
print_data_info(processed_data)