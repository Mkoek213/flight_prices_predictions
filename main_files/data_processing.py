from main_files import flight_data
from main_files import plt
from main_files import sns

print(flight_data.info())
print(flight_data.head()) #first few rows
print(flight_data.describe(include='all')) #some info about data
print(flight_data.isnull().sum()) #no nulls, we're good

def plot_flight_data():
    flight_data_process_copy = flight_data.copy()
    # plot only columns which dtype is not 'Object'
    columns_to_plot = [column for column in flight_data_process_copy.columns if
                       flight_data_process_copy[column].dtypes != 'O']

    figure, ax = plt.subplots(3, 4, figsize=(12, 9))
    ax = ax.flatten()

    for i, column in enumerate(columns_to_plot):
        sns.histplot(flight_data_process_copy[column], kde=True, stat='density', bins=30, ax=ax[i])

    for i in range(len(columns_to_plot), 12):
        ax[i].axis('off')

    plt.tight_layout()
    plt.savefig('plot.png')
    plt.close()


plot_flight_data()
