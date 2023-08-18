import numpy as np
import pandas as pd
from analysis_power import read_npz_files
import matplotlib.pyplot as plt
import os

def plot_results(data, title,xsticks):
    columns = ['beta', 'Imputer_Median', 'Imputer_LinearRegression',  'Imputer_GradientBoosting', 'Imputer_Oracle']

    df = pd.DataFrame(data, columns=columns)

    plt.figure(figsize=(10, 6))

    colors = {'Median': 'blue', 'LinearRegression': 'red', 'GradientBoosting': 'orange', 'Oracle':'purple'}
    linestyles = {'Imputer': '-'}

    for col in columns[1:]:
        method = col.split('_')[1]
        dataset = col.split('_')[0]
        linestyle = linestyles[dataset]
        plt.plot(df['beta'], df[col], marker='o', label=method, color=colors[method], linestyle=linestyle)
        
    plt.xlabel('Beta')
    plt.ylabel('Power')
    plt.title(title)
    plt.legend()
    plt.grid()
    
    # Setting y-axis ticks with custom intervals
    y_ticks = [i/100.0 for i in range(0, 105, 20)]  # Starts from 0, ends at 1.05, with an interval of 0.05
    y_ticks.append(0.05)
    plt.yticks(y_ticks)

    X_ticks = xsticks
    plt.xticks(X_ticks)

    #plt.show()
    if not os.path.exists("pic"):
        os.makedirs("pic")

    plt.savefig("pic/"+title+".png", format='png', dpi=600) 

def plot(range,dir,title, small_size, xsticks):
    print(range)
    data = []
    for coef in range:
        row_power = [coef]
        print("Result/%s/%f" % (dir,coef))
        for directory in ["Result/%s/%f" % (dir,coef)]:
            results = read_npz_files(directory,small_size=small_size)
            if small_size:
                row_power.extend([results['median_power'], results['lr_power'], results['xgboost_power'],results['oracle_power']])
            else:
                row_power.extend([results['median_power'], results['lr_power'],results['lightGBM_power'], results['oracle_power']])
        data.append(row_power)
    plot_results(data,title,xsticks) 

def main():

    plot(np.arange(0.0, 0.72, 0.12),"HPC_power__unobserved" + "_multi","Size-100, Multi-missing, U", True, np.arange(0.0, 0.72, 0.12))

    plot(np.arange(0.0, 0.18, 0.03),"HPC_power_1000_unobserved" + "_multi","Size-1000, Multi-missing, U", False, np.arange(0.0, 0.18, 0.03))

main()
