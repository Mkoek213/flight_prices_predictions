from main_files import flight_data
from main_files import plt
from main_files import sns
from main_files import np
from main_files import pd
import io
import sys
from datetime import datetime

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
    df_encoded_source = pd.get_dummies(flight_data_processed['Source'], prefix='Source').astype(int)
    flight_data_processed = pd.concat([flight_data_processed, df_encoded_source], axis=1)
    flight_data_processed.drop('Source', axis=1, inplace=True)

    # One hot encoding 'Destination' parameter
    df_encoded_destination = pd.get_dummies(flight_data_processed['Destination'], prefix='Destination').astype(int)
    flight_data_processed = pd.concat([flight_data_processed, df_encoded_destination], axis=1)
    flight_data_processed.drop('Destination', axis=1, inplace=True)

    return flight_data_processed

def categorize_hour(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    elif 21 <= hour <= 23 or 0 <= hour < 5:
        return 'Night'
    else:
        return 'Invalid Hour'

def calculate_day_of_week(row):
    date = datetime(year=2019, month=row['Month'], day=row['Date'])
    return date.strftime('%A')

def process_time_data(database):
    flight_data_processed = database.copy()
    # Counting total duration time
    flight_data_processed['Total_Duration_min'] = flight_data_processed['Duration_hours'] * 24 + flight_data_processed['Duration_min']
    flight_data_processed.drop('Duration_hours', axis=1, inplace=True)
    flight_data_processed.drop('Duration_min', axis=1, inplace=True)

    # Categorizing 'Dep_hours' and 'Arrival_hours'
    flight_data_processed['Departure_Time_categorized'] = flight_data_processed['Dep_hours'].apply(categorize_hour)
    flight_data_processed['Arrival_Time_categorized'] = flight_data_processed['Arrival_hours'].apply(categorize_hour)
    flight_data_processed.drop(columns=['Dep_hours', 'Arrival_hours', 'Dep_min', 'Arrival_min'], inplace=True, axis=1)

    # One hot encoding 'Departure_Time_categorized'
    df_encoded_dep = pd.get_dummies(flight_data_processed['Departure_Time_categorized'], prefix='Departure').astype(int)
    flight_data_processed = pd.concat([flight_data_processed, df_encoded_dep], axis=1)
    flight_data_processed.drop('Departure_Time_categorized', axis=1, inplace=True)

    # One hot encoding 'Arrival_Time_categorized'
    df_encoded_arrival = pd.get_dummies(flight_data_processed['Arrival_Time_categorized'], prefix='Arrival').astype(int)
    flight_data_processed = pd.concat([flight_data_processed, df_encoded_arrival], axis=1)
    flight_data_processed.drop('Arrival_Time_categorized', axis=1, inplace=True)

    # Mapping month
    month_mapping = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April',
        5: 'May', 6: 'June', 7: 'July', 8: 'August',
        9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }

    flight_data_processed['Month_Name'] = flight_data_processed['Month'].map(month_mapping)

    # One hot encoding 'Month_Name'
    df_encoded_month = pd.get_dummies(flight_data_processed['Month_Name'], prefix='Month').astype(int)
    flight_data_processed = pd.concat([flight_data_processed, df_encoded_month], axis=1)
    flight_data_processed.drop('Month_Name', axis=1, inplace=True)

    # Calculate day of week
    flight_data_processed['Day_of_Week'] = flight_data_processed.apply(calculate_day_of_week, axis=1)

    # One hot encoding 'Day_of_Week'
    df_encoded_day = pd.get_dummies(flight_data_processed['Day_of_Week'], prefix='Day').astype(int)
    flight_data_processed = pd.concat([flight_data_processed, df_encoded_day], axis=1)
    flight_data_processed.drop(columns=['Day_of_Week', 'Date', 'Month'], axis=1, inplace=True)
    return flight_data_processed

def categorize_day(day):
    if 5 <= day < 12:
        return 'first_week'
    elif 12 <= day < 17:
        return 'second_week'
    elif 17 <= day < 21:
        return 'third_week'
    elif 21 <= day <= 23:
        return 'fourth_week'
    else:
        return 'Invalid Hour'




processed_data = one_hot_encoding_category_data(flight_data)
# print_data_info(processed_data)
# plot_for_outliers(flight_data)
# plot_correlation_heatmap(flight_data)
# plot_distribution_info(flight_data)
# plot_category_data_distribution(flight_data)
processed_data2 = process_time_data(processed_data)
print_data_info(processed_data2)

print(len(processed_data2.columns))