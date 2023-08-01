import numpy as np
import pandas as pd
from analysis_power import read_npz_files
import matplotlib.pyplot as plt

def plot_results1(data, title):
    columns = ['beta', 'MICE_Median', 'MICE_Linear',  'MICE_LightGBM']

    df = pd.DataFrame(data, columns=columns)

    plt.figure(figsize=(10, 6))

    colors = {'Median': 'blue', 'Linear': 'red', 'LightGBM': 'orange' }
    linestyles = {'MICE': '-'}

    for col in columns[1:]:
        method = col.split('_')[1]
        dataset = col.split('_')[0]
        linestyle = linestyles[dataset]
        plt.plot(df['beta'], df[col], marker='o', label=col, color=colors[method], linestyle=linestyle)
        
    plt.xlabel('Beta')
    plt.ylabel('Power')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def plot_results(data, title):
    columns = ['beta', 'Imputer_Median', 'Imputer_Linear',  'Imputer_LightGBM', 'Imputer_Oracle']

    df = pd.DataFrame(data, columns=columns)

    plt.figure(figsize=(10, 6))

    colors = {'Median': 'blue', 'Linear': 'red', 'LightGBM': 'orange',  'Oracle':'purple'}
    linestyles = {'Imputer': '-'}

    for col in columns[1:]:
        method = col.split('_')[1]
        dataset = col.split('_')[0]
        linestyle = linestyles[dataset]
        plt.plot(df['beta'], df[col], marker='o', label=col, color=colors[method], linestyle=linestyle)
        
    plt.xlabel('Beta')
    plt.ylabel('Power')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def plot(range,dir,title):
    print(range)
    data = []
    for coef in range:
        row_power = [coef]
        print("Result/%s/%f" % (dir,coef))
        for directory in ["Result/%s/%f" % (dir,coef)]:
            results = read_npz_files(directory)
            row_power.extend([results['median_power'], results['lr_power'], results['lightGBM_power'],results['oracle_power']])
        data.append(row_power)
    print(data)
    plot_results(data,title) 


def main():

    plot(np.arange(0,1.5,0.3),"HPC_power_50_unobserved_linearZ_linearX" + "_single","Size-50, linearZ_linearX, U")

    plot(np.arange(0.0,0.4,0.08),"HPC_power_1000_unobserved_linearZ_linearX" + "_single","Size-1000, linearZ_linearX, U")

    plot(np.arange(0.0,5,1),"HPC_power_50_unobserved_linearZ_nonlinearX" + "_single","Size-50, linearZ_nonlinearX, U")

    plot(np.arange(0.0,0.80,0.16),"HPC_power_1000_unobserved_linearZ_nonlinearX" + "_single","Size-1000, linearZ_nonlinearX, U")

    plot(np.arange(0.0,1.5,0.25),"HPC_power_50_unobserved_nonlinearZ_nonlinearX" + "_single","Size-50, nonlinearZ_nonlinearX, U")

    plot(np.arange(0.0,0.3 ,0.05),"HPC_power_1000_unobserved_nonlinearZ_nonlinearX" + "_single","Size-1000, nonlinearZ_nonlinearX, U")

main()
