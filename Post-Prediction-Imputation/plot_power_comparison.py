import numpy as np
import pandas as pd
from analysis_power import read_npz_files
from Visulization import plot_all, plot_size, plot_types, plot_unobserved


def main(type):
    data = []
    data_with_U = []

    for coef in np.arange(0.01,0.3,0.01):
        row = [coef]
        row_with_U = [coef]
        for directory in ["Result/HPC_power_1000_%s/%f" % (type,coef),
                          "Result/HPC_power_2000_%s/%f" % (type,coef)]:
            results = read_npz_files(directory)
            row.extend([results['median'], results['lr'], results['xgboost'], results['oracle']])
            
            results_with_U = read_npz_files(directory.replace("HPC_power_", "HPC_power_unobserved_"))
            row_with_U.extend([results_with_U['median'], results_with_U['lr'], results_with_U['xgboost'], results_with_U['oracle']])

        data.append(row)
        data_with_U.append(row_with_U)

    print(data)
    print(data_with_U)
    plot(data, data_with_U)



def plot(data, data_with_U):
    plot_all.plot_results(data, data_with_U)
    plot_size.plot_results(data)
    plot_types.plot_results(data, data_with_U)
    plot_unobserved.plot_results(data, data_with_U)

main("single")
#main("multi")


