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
    columns = ['beta', 'Imputer_Median', 'Imputer_Linear',  'Imputer_LightGBM', 'Imputer_xgboost', 'Imputer_Oracle']

    df = pd.DataFrame(data, columns=columns)

    plt.figure(figsize=(10, 6))

    colors = {'Median': 'blue', 'Linear': 'red', 'LightGBM': 'orange', 'xgboost': 'green', 'Oracle':'purple'}
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
            row_power.extend([results['median_power'], results['lr_power'], results['lightGBM_power'], results['xgboost_power'],results['oracle_power']])
        data.append(row_power)
    print(data)
    plot_results(data,title) 


def main():

    plot(np.arange(0.0,0.3,0.05),"HPC_power_1000_unobserved_interference_adjutment_0" + "_single","L-100,Size-1000,no-adjustment,U")
    plot(np.arange(0.0,1.5,0.25),"HPC_power_100_unobserved_interference_adjutment_0" + "_single","L-100,Size-50,no-adjustment,U")
    
    plot(np.arange(0.0,0.3,0.05),"HPC_power_1000_unobserved_interference_adjutment_1" + "_single","L-100,Size-1000,cov-adjustment-LR,U")
    plot(np.arange(0.0,1.5,0.25),"HPC_power_100_unobserved_interference_adjutment_1" + "_single","L-100,Size-50,cov-adjustment-LR,U")
    
    plot(np.arange(0.0,0.3,0.05),"HPC_power_1000_unobserved_interference_adjutment_2" + "_single","L-100,Size-1000,cov-adjustment-original,U")
    plot(np.arange(0.0,1.5,0.25),"HPC_power_100_unobserved_interference_adjutment_2" + "_single","L-100,Size-50,cov-adjustment-original,U")

    exit()
    #plot(np.arange(0.0,3.1,0.5),"HPC_power_50_linearZ_linearX" + "_single","Size-50, linearZ_linearX, No U")
    plot(np.arange(0.0,3.1,0.5),"HPC_power_50_unobserved_linearZ_linearX" + "_single","Size-50, linearZ_linearX, U")

    #plot(np.arange(0.0,0.41,0.08),"HPC_power_2000_linearZ_linearX" + "_single","Size-2000, linearZ_linearX, No U")
    plot(np.arange(0.0,0.41,0.08),"HPC_power_2000_unobserved_linearZ_linearX" + "_single","Size-2000, linearZ_linearX, U")

    #plot(np.arange(0.0,10.1,2),"HPC_power_50_linearZ_nonlinearX" + "_single","Size-50, linearZ_nonlinearX, No U")
    plot(np.arange(0.0,10.1,2),"HPC_power_50_unobserved_linearZ_nonlinearX" + "_single","Size-50, linearZ_nonlinearX, U")

    #plot(np.arange(0.0,0.81,0.15),"HPC_power_2000_linearZ_nonlinearX" + "_single","Size-2000, linearZ_nonlinearX, No U")
    plot(np.arange(0.0,0.81,0.15),"HPC_power_2000_unobserved_linearZ_nonlinearX" + "_single","Size-2000, linearZ_nonlinearX, U")

    #plot(np.arange(0.0,6.1,1),"HPC_power_50_nonlinearZ_nonlinearX" + "_single","Size-50, nonlinearZ_nonlinearX, No U")
    plot(np.arange(0.0,6.1,1),"HPC_power_50_unobserved_nonlinearZ_nonlinearX" + "_single","Size-50, nonlinearZ_nonlinearX, U")

    #plot(np.arange(0.0,0.31,0.06),"HPC_power_2000_nonlinearZ_nonlinearX" + "_single","Size-2000, nonlinearZ_nonlinearX, No U")
    plot(np.arange(0.0,0.31,0.06),"HPC_power_2000_unobserved_nonlinearZ_nonlinearX" + "_single","Size-2000, nonlinearZ_nonlinearX, U")

main()
