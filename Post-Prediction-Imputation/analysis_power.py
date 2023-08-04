import numpy as np
import os

def read_npz_files(directory):
    summed_p_values_median = None
    summed_p_values_LR = None
    summed_p_values_lightGBM = None
    summed_p_values_oracle = None
    summed_p_values_xgboost = None

    N = int(len(os.listdir(directory)) / 5)

    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            filepath = os.path.join(directory, filename)
            p_values = np.load(filepath)

            if "p_values_median" in filename:
                if summed_p_values_median is None:
                    summed_p_values_median = (p_values<= 0.05).astype(int)
                else:
                    summed_p_values_median += (p_values<= 0.05).astype(int)
            elif "p_values_LR" in filename:
                if summed_p_values_LR is None:
                    summed_p_values_LR = (p_values<= 0.05).astype(int)
                else:
                    summed_p_values_LR += (p_values<= 0.05).astype(int)
            elif "p_values_lightGBM" in filename:
                if summed_p_values_lightGBM is None:
                    summed_p_values_lightGBM = (p_values<= 0.05).astype(int)
                else:
                    summed_p_values_lightGBM += (p_values<= 0.05).astype(int)
            elif "p_values_xgboost" in filename:
                if summed_p_values_xgboost is None:
                    summed_p_values_xgboost = (p_values<= 0.05).astype(int)
                else:
                    summed_p_values_xgboost += (p_values<= 0.05).astype(int)
            elif "p_values_oracle" in filename:
                if summed_p_values_oracle is None:
                    summed_p_values_oracle = (p_values<= 0.05).astype(int)
                else:
                    summed_p_values_oracle += (p_values<= 0.05).astype(int)

    results = {
        'median_power': summed_p_values_median[0] / N,
        'median_corr': summed_p_values_median[2] / N,
        'lr_power': summed_p_values_LR[0] / N,
        'lr_corr': summed_p_values_LR[2] / N,
        'lightGBM_power': summed_p_values_lightGBM[0] / N,
        'lightGBM_corr': summed_p_values_lightGBM[2] / N,
        'oracle_power': summed_p_values_oracle[0] / N,
        'oracle_corr': summed_p_values_oracle[2] / N,
        'xgboost_power': summed_p_values_xgboost[0] / N,
        'xgboost_corr': summed_p_values_xgboost[2] / N,
    }
    return results

