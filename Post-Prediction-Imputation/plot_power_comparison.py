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
        'Imputer_PREP-GBM': 'green', 
        'Imputer_Oracle': 'purple',
        "Imputer_Median": "blue",
        "Imputer_PREP-RidgeReg": "red",
        "Imputer_GBM-adjusted": 'green', 
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
        plt.plot(df['beta'], df[col], marker='o', color=colors[col], linestyle=linestyles[col], linewidth=2)

    plt.xlabel(r'$\beta$',fontsize=30)
    plt.ylabel('Power',fontsize=30)
    plt.grid()

    if not os.path.exists("pic"):
        os.makedirs("pic")

    plt.xticks(xsticks)
    y_ticks = [i / 100.0 for i in range(25, 105, 25)]
    y_ticks.append(0.05)
    plt.yticks(y_ticks)
    plt.tick_params(axis='both', which='major', labelsize=25)


    plt.savefig("pic/" + title + ".svg", format='svg')

def main():
    Power_data = []
    Power_data_small = []
    plot_results(Power_data,  "Size-1000, Single: Covariance Adjusted, ", np.arange(0.0,0.3 ,0.05)) 

    for coef in np.arange(0.0,0.3 ,0.05):
        row_power = [coef]
        for directory in [ "Result/HPC_power_1000_unobserved_interference_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=False)
            row_power.extend([ results['lightGBM_power'], results['oracle_power'], results['median_power'], results['lr_power']])
        for directory in ["Result/HPC_power_1000_unobserved_interference_adjusted_1_single/%f" % (coef)]:
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
        for directory in [ "Result/HPC_power_50_unobserved_interference_adjusted_1_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=True)
            row_power_small.extend([results['xgboost_power'], results['oracle_power'], results['median_power'], results['lr_power']])
        Power_data_small.append(row_power_small)
    print(Power_data)
    plot_results(Power_data_small, "Size-50, Single: Covariance Adjusted, ", np.arange(0.0,1.2,0.2))   
     

main()

# Create a new figure for the legend
fig_leg, ax_leg = plt.subplots(figsize=(6, 1))  # Adjust the size as needed
ax_leg.axis('off')

# Create the legend
custom_lines_types = [
    Line2D([0], [0], color='blue', lw=2),
    Line2D([0], [0], color='red', lw=2),
    Line2D([0], [0], color='green', lw=2),
    Line2D([0], [0], color='purple', lw=2)
]
custom_lines_adjustment = [
    Line2D([0], [0], color='black', lw=2, linestyle='--'),
    Line2D([0], [0], color='black', lw=2, linestyle='-')
]
legend = ax_leg.legend(custom_lines_types + custom_lines_adjustment, 
                       ['Median', 'PREP-RidgeReg', 'PREP-GBM', 'Oracle', 'Adjusted', 'Original'], 
                       loc='center', 
                       ncol=3,  # Adjust as needed
                       fontsize='large')  # Adjust as needed

# Save the legend
fig_leg.savefig('pic/legend.svg', format='svg', bbox_inches='tight')




# Create a new figure for the custom_lines_types legend
fig_leg1, ax_leg1 = plt.subplots(figsize=(12, 1))  # Adjust the size as needed
ax_leg1.axis('off')

# Create the legend for custom_lines_types
legend1 = ax_leg1.legend(custom_lines_types, 
                          ['Method 1', 'Method 2', 'Method 3', 'Method 4'], 
                          loc='center', 
                          ncol=4,  # Set to 4 to make all items appear in a single line
                          fontsize='large')  # Adjust as needed

# Save the legend
fig_leg1.savefig('pic/legend_custom_lines_types.svg', format='svg', bbox_inches='tight')


# Create a new figure for the custom_lines_adjustment legend
fig_leg2, ax_leg2 = plt.subplots(figsize=(2, 1))  # Adjust the size as needed
ax_leg2.axis('off')

# Create the legend for custom_lines_adjustment
legend2 = ax_leg2.legend(custom_lines_adjustment, 
                          ['Adjusted', 'Original'], 
                          loc='center', 
                          ncol=2,  # Adjust as needed
                          fontsize='large')  # Adjust as needed

# Save the legend
fig_leg2.savefig('pic/legend_custom_lines_adjustment.svg', format='svg', bbox_inches='tight')



