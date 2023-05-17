import numpy as np
import os

def read_npz_files(directory):
    summed_p_values_median = None
    summed_p_values_LR = None
    summed_p_values_xgboost = None
    summed_p_values_oracle = None

    N = int(len(os.listdir(directory)) / 4)

    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            filepath = os.path.join(directory, filename)
            p_values = np.load(filepath)

            if "p_values_median" in filename:
                if summed_p_values_median is None:
                    summed_p_values_median = p_values
                else:
                    summed_p_values_median += p_values
            elif "p_values_LR" in filename:
                if summed_p_values_LR is None:
                    summed_p_values_LR = p_values
                else:
                    summed_p_values_LR += p_values
            elif "p_values_xgboost" in filename:
                if summed_p_values_xgboost is None:
                    summed_p_values_xgboost = p_values
                else:
                    summed_p_values_xgboost += p_values
            elif "p_values_oracle" in filename:
                if summed_p_values_oracle is None:
                    summed_p_values_oracle = p_values
                else:
                    summed_p_values_oracle += p_values

    results = {
        'median_power': summed_p_values_median[1] / N,
        'median_corr': summed_p_values_median[2] / N,
        'lr_power': summed_p_values_LR[1] / N,
        'lr_corr': summed_p_values_LR[2] / N,
        'xgboost_power': summed_p_values_xgboost[1] / N,
        'xgboost_corr': summed_p_values_xgboost[2] / N,
        'oracle_power': summed_p_values_oracle[1] / N,
        'oracle_corr': summed_p_values_oracle[2] / N
    }
    return results

