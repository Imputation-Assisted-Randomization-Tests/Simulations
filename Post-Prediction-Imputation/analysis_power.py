import numpy as np
import os

def read_and_print_npz_files(directory, file):
    
    file.write("Analysis of : " + directory + "\n")

    summed_p_values_median = None
    summed_p_values_LR = None
    summed_p_values_xgboost = None
    summed_p_values_oracle = None

    N = int(len(os.listdir(directory)) / 4)
    # Loop through all the files in the directory
    for filename in os.listdir(directory):

        # Check if the file is a numpy file
        if filename.endswith(".npy"):
            # Load the numpy array from the file
            filepath = os.path.join(directory, filename)
            p_values = np.load(filepath)

            # Add the array to the running sum based on the filename
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

    file.write("Mean p-values for Median Imputer:\n")
    file.write(str(summed_p_values_median/N) + "\n")
    file.write("Mean p-values for LR Imputer:\n")
    file.write(str(summed_p_values_LR/N) + "\n")
    file.write("Mean p-values for XGBoost Imputer:\n")
    file.write(str(summed_p_values_xgboost/N) + "\n")
    file.write("Mean p-values for Oracle:\n")
    file.write(str(summed_p_values_oracle/N) + "\n")

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
        'median': summed_p_values_median[8] / N,
        'lr': summed_p_values_LR[8] / N,
        'xgboost': summed_p_values_xgboost[8] / N,
        'oracle': summed_p_values_oracle[8] / N
    }
    return results

def main():
    for coef in np.arange(0.02, 0.2, 0.02):
        with open("power.result", "a") as file:
            file.write("beta: " + str(coef) + "\n")
            read_and_print_npz_files("Result/HPC_power_unobserved_1000_single/%f" % coef, file)
            file.write("\n")
            read_and_print_npz_files("Result/HPC_power_1000_single/%f" % coef, file)
            file.write("\n")
            read_and_print_npz_files("Result/HPC_power_unobserved_2000_single/%f" % coef, file)
            file.write("\n")
            read_and_print_npz_files("Result/HPC_power_2000_single/%f" % coef, file)
            file.write("\n")
            read_and_print_npz_files("Result/HPC_power_unobserved_2000_multi/%f" % coef, file)
            file.write("\n")
            read_and_print_npz_files("Result/HPC_power_2000_multi/%f" % coef, file)
            file.write("\n")
            read_and_print_npz_files("Result/HPC_power_unobserved_1000_multi/%f" % coef, file)
            file.write("\n")
            read_and_print_npz_files("Result/HPC_power_1000_multi/%f" % coef, file)
            file.write("\n\n")

#main()

