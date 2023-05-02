import numpy as np
import pandas as pd
from analysis_power import read_npz_files
from Visulization import plot_all, plot_size, plot_types, plot_unobserved


def main(type):
    data = []

    for coef in np.arange(0.02, 0.2, 0.02):
        row = [coef]
        for directory in ["Result/HPC_power_1000_%s/%f" % (type,coef),
                          "Result/HPC_power_2000_%s/%f" % (type,coef)]:
            results = read_npz_files(directory)
            row.extend([results['median'], results['lr'], results['xgboost'], results['oracle']])
        data.append(row)
    
    plot_size.plot_results(data)

    

main("single")
#main("multi")


