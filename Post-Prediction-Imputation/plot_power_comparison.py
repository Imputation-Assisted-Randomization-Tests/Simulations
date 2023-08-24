import numpy as np
import pandas as pd
from analysis_power import read_npz_files
import matplotlib.pyplot as plt
import os

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_results(data, title, xsticks):
    plt.clf()

    columns = ['beta', 'Imputer_PREP-GBM', 'Imputer_Oracle', "Imputer_Median", 
               "Imputer_PREP-RidgeReg", "Imputer_GBM-adjusted", "Imputer_Oracle-adjusted", 
               "Imputer_Median-adjusted", "Imputer_LR-adjusted"]

    df = pd.DataFrame(data, columns=columns)

    plt.figure(figsize=(10, 6))

    colors = {
        'Imputer_PREP-GBM': 'orange', 
        'Imputer_Oracle': 'purple',
        "Imputer_Median": "blue",
        "Imputer_PREP-RidgeReg": "red",
        "Imputer_GBM-adjusted": 'orange', 
        "Imputer_Oracle-adjusted": 'purple', 
        "Imputer_Median-adjusted": 'blue', 
        "Imputer_LR-adjusted": 'red'
    }

    linestyles = {
        'Imputer_PREP-GBM': '-', 
        'Imputer_Oracle': '-', 
        "Imputer_Median": '-',
        "Imputer_PREP-RidgeReg": '-', 
        "Imputer_GBM-adjusted": '--', 
        "Imputer_Oracle-adjusted": '--',
        "Imputer_Median-adjusted": '--',
        "Imputer_LR-adjusted": '--'
    }

    for col in columns[1:]:
        plt.plot(df['beta'], df[col], marker='o', color=colors[col], linestyle=linestyles[col])

    custom_lines_types = [
        Line2D([0], [0], color='blue', lw=2),
        Line2D([0], [0], color='red', lw=2),
        Line2D([0], [0], color='orange', lw=2),
        Line2D([0], [0], color='purple', lw=2)
    ]
    legend1 = plt.legend(custom_lines_types, ['Median', 'PREP-RidgeReg', 'PREP-GBM', 'Oracle'], loc='upper left')
    plt.gca().add_artist(legend1)

    custom_lines_adjustment = [
        Line2D([0], [0], color='black', lw=2, linestyle='--'),
        Line2D([0], [0], color='black', lw=2, linestyle='-')
    ]
    plt.legend(custom_lines_adjustment, ['Adjusted', 'Original'], title='Covariate Adjustment', loc='upper left', bbox_to_anchor=(0, 0.7))


    plt.xlabel(r'$\beta$')
    plt.ylabel('Power')
    plt.grid()

    if not os.path.exists("pic"):
        os.makedirs("pic")

    plt.xticks(xsticks)
    y_ticks = [i / 100.0 for i in range(0, 105, 20)]
    y_ticks.append(0.05)
    plt.yticks(y_ticks)

    plt.savefig(f"pic/{title}.png", format='png', dpi=600)

# Example usage:
# data = your_data_here
# title = 'Your Title'
# xsticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
# plot_results(data, title, xsticks)

def main():
    Power_data = []
    Power_data_small = []
    plot_results(Power_data,  "Size-1000, Single: Covariance Adjusted, ", np.arange(0.0,0.3 ,0.05)) 

    for coef in np.arange(0.0,0.3 ,0.05):
        row_power = [coef]
        for directory in [ "Result/HPC_power_1000_unobserved_interference_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=False)
            row_power.extend([ results['lightGBM_power'], results['oracle_power'], results['median_power'], results['lr_power']])
        for directory in ["Result/HPC_power_1000_unobserved_interference_adjusted_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=False)
            row_power.extend([ results['lightGBM_power'],results['oracle_power'], results['median_power'], results['lr_power'] ])
        Power_data.append(row_power)

    print(Power_data)
    plot_results(Power_data,  "Size-1000, Single: Covariance Adjusted, ", np.arange(0.0,0.3 ,0.05)) 

    for coef in np.arange(0.0,1.2,0.2):
        row_power_small = [coef]
        for directory in ["Result/HPC_power_50_unobserved_interference_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=True)
            row_power_small.extend([results['xgboost_power'], results['oracle_power'], results['median_power'], results['lr_power']])
        for directory in [ "Result/HPC_power_50_unobserved_interference_adjusted_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=True)
            row_power_small.extend([results['xgboost_power'], results['oracle_power'], results['median_power'], results['lr_power']])
        Power_data_small.append(row_power_small)
    print(Power_data)
    plot_results(Power_data_small, "Size-50, Single: Covariance Adjusted, ", np.arange(0.0,1.2,0.2))   
     

main()


