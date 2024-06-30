from main_files import flight_data
from main_files import plt
from main_files import sns
from main_files import np

print(flight_data.info())
print(flight_data.head()) #first few rows
print(flight_data.describe(include='all')) #some info about data
print(flight_data.isnull().sum()) #no nulls, we're good

def plot_flight_data():
    flight_data_plot_copy = flight_data.copy()
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
    plt.savefig('plot.png')
    plt.close()


plot_flight_data()
