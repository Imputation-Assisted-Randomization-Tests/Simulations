import os
import numpy as np
import matplotlib.pyplot as plt
import scipy

threshold_005 = 0.05
threshold_010 = 0.10
threshold_020 = 0.20

def proportions_below_threshold(p_values, threshold):
    count = sum(1 for p in p_values if p < threshold)
    proportion = count / len(p_values)
    return proportion

def read_and_print_npz_files(directory, file):

    file.write("Analysis of : " + directory + "\n")

    summed_p_values_median = None
    summed_p_values_LR = None
    summed_p_values_lightGBM = None
    summed_p_values_oracle = None

    p_values_median = []
    p_values_LR = []
    p_values_lightGBM = []
    p_values_oracle = []

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
                p_values_median.append(list(p_values))
            elif "p_values_LR" in filename:
                if summed_p_values_LR is None:
                    summed_p_values_LR = p_values
                else:
                    summed_p_values_LR += p_values
                p_values_LR.append(list(p_values))
            elif "p_values_lightGBM" in filename:
                if summed_p_values_lightGBM is None:
                    summed_p_values_lightGBM = p_values
                else:
                    summed_p_values_lightGBM += p_values
                p_values_lightGBM.append(list(p_values))
            elif "p_values_oracle" in filename:
                if summed_p_values_oracle is None:
                    summed_p_values_oracle = p_values
                else:
                    summed_p_values_oracle += p_values
                p_values_oracle.append(list(p_values))


    file.write("Mean p-values for Median Imputer:\n")
    file.write("Power: " + str(summed_p_values_median/N) + "\n")
    file.write("Mean p-values for LR Imputer:\n")
    file.write("Power: " + str(summed_p_values_LR/N) + "\n")
    file.write("Mean p-values for lightGBM Imputer:\n")
    file.write("Power: " + str(summed_p_values_lightGBM/N) + "\n")
    file.write("Mean p-values for Oracle Imputer:\n")
    file.write("Power: " + str(summed_p_values_oracle/N) + "\n")

    file.write("Plotting the distribution of the first 6 p-values for each imputer\n")

    file.write("Median Imputer\n")
    plot_p_values_distribution(p_values_median, "Median Imputer", file)
    file.write("LR Imputer\n")
    plot_p_values_distribution(p_values_LR, "LR Imputer", file)
    file.write("LightGBM Imputer\n")
    plot_p_values_distribution(p_values_lightGBM, "LightGBM Imputer", file)
    file.write("Oracle Imputer\n")
    plot_p_values_distribution(p_values_oracle, "Oracle Imputer", file)
    file.write("\n")

def plot_p_values_distribution(p_values, imputer_name, file):
    p_values = np.array(p_values)
    fig, axs = plt.subplots(1, 6, figsize=(12, 4), tight_layout=True)

    for i in range(0, 1):
        axs[i].hist(p_values[:, i])
        axs[i].set_title(f"p-value {i + 1}")
        file.write(str(scipy.stats.kstest(p_values[:, i], 'uniform')) + "\n")
        proportion_below_005 = proportions_below_threshold(p_values[:, i], threshold_005)
        file.write(f"Proportion of p-values below {threshold_005}: {proportion_below_005:.4f}\n")
        file.write(f"Proportion of p-values below {threshold_020}: {proportions_below_threshold(p_values[:, i], threshold_020):.4f}\n")
        continue 


        # Histogram
        plt.figure(figsize=(8, 6))
        plt.hist(p_values[:,i], bins=20, color='g', alpha=0.7)
        plt.xlabel('P-values')
        plt.ylabel('Frequency')
        plt.title('Histogram of P-values')
        plt.show()

        # You can create other visualizations as well, like box plots, violin plots, etc.
        # For example, a box plot of the p-values:
        plt.figure(figsize=(8, 6))
        plt.boxplot(p_values[:,i], vert=False)
        plt.xlabel('P-values')
        plt.title('Box Plot of P-values')
        plt.show()

with open("level.result", "w") as file:
    read_and_print_npz_files('Result/HPC_power_1000_unobserved_linearZ_linearX_single/0.000000', file)
    read_and_print_npz_files('Result/HPC_power_1000_unobserved_linearZ_nonlinearX_single/0.000000', file)
    read_and_print_npz_files('Result/HPC_power_1000_unobserved_nonlinearZ_nonlinearX_single/0.000000', file)

