import pandas as pd
import matplotlib.pyplot as plt

def plot_results2(data, data_LR_adjusted, data_xgboost_adjusted):
    data =  [[row[0], row[1], row[2], row[3], row[4]] for row in data]
    data_LR_adjusted =  [[row[0], row[1], row[2], row[3], row[4]] for row in data_LR_adjusted]
    data_xgboost_adjusted =  [[row[0], row[1], row[2], row[3], row[4]] for row in data_xgboost_adjusted]
    
    columns = [['beta', 'Median', 'Linear', 'XGBoost', 'Oracle'], ['beta', 'Median_LR', 'Linear_LR', 'XGBoost_LR', 'Oracle_LR'], ['beta', 'Median_XGBoost', 'Linear_XGBoost', 'XGBoost_XGBoost', 'Oracle_XGBoost']]

    df = pd.DataFrame(data, columns=columns[0])
    df_LR_adjusted = pd.DataFrame(data_LR_adjusted, columns=columns[1])
    df_xgboost_adjusted = pd.DataFrame(data_xgboost_adjusted, columns=columns[2])

    colors = {'Median': 'blue', 'Linear': 'red', 'XGBoost': 'green', 'Oracle': 'purple'}

    dfs = [df, df_LR_adjusted, df_xgboost_adjusted]
    titles = ['Performance of Original Data', 'Performance of LR Adjusted Data', 'Performance of XGBoost Adjusted Data']

    for i in range(3):
        plt.figure(figsize=(12, 8))
        for col in dfs[i].columns[1:]:
            method = col.split('_')[0]
            plt.plot(dfs[i]['beta'], dfs[i][col], marker='o', label=col, color=colors[method])
        plt.xlabel('Beta')
        plt.ylabel('Absolute Error')
        plt.title(titles[i])
        plt.legend()
        plt.grid()
        plt.show()

def plot_results(data, data_LR_adjusted, data_xgboost_adjusted, data_lightGBM_adjusted):
    data = [[row[0], row[1], row[2], row[3], row[4], row[5]] for row in data]
    data_LR_adjusted = [[row[0], row[1], row[2], row[3], row[4], row[5]] for row in data_LR_adjusted]
    data_xgboost_adjusted = [[row[0], row[1], row[2], row[3], row[4], row[5]] for row in data_xgboost_adjusted]
    data_lightGBM_adjusted = [[row[0], row[1], row[2], row[3], row[4], row[5]] for row in data_lightGBM_adjusted]
    
    columns = ['beta', 'Median', 'Linear', 'XGBoost', 'Oracle', 'lightGBM']
    columns_LR = ['beta', 'Median_LR', 'Linear_LR', 'XGBoost_LR', 'Oracle_LR', 'lightGBM_LR']
    columns_XGBoost = ['beta', 'Median_XGBoost', 'Linear_XGBoost', 'XGBoost_XGBoost', 'Oracle_XGBoost', 'lightGBM_XGBoost']
    columns_lightGBM = ['beta', 'Median_lightGBM', 'Linear_lightGBM', 'XGBoost_lightGBM', 'Oracle_lightGBM', 'lightGBM_lightGBM']

    
    df = pd.DataFrame(data, columns=columns)
    df_LR = pd.DataFrame(data_LR_adjusted, columns=columns_LR)
    df_XGBoost = pd.DataFrame(data_xgboost_adjusted, columns=columns_XGBoost)
    df_lightGBM = pd.DataFrame(data_lightGBM_adjusted, columns=columns_lightGBM)

    linestyles = ['-', '--', '-.', ':']
    datasets = [df, df_LR, df_XGBoost, df_lightGBM]
    labels = ['Original', 'LR Adjusted', 'XGBoost Adjusted', 'lightGBM Adjusted']

    methods = ['Median', 'Linear', 'XGBoost', 'Oracle', 'lightGBM']
    adjusted_methods = ['_LR', '_XGBoost', '_lightGBM']

    for method in methods:
        plt.figure(figsize=(10, 6))
        
        # Original data
        plt.plot(df['beta'], df[method], linestyle=linestyles[0], marker='o', label=f'{method} {labels[0]}')
        
        # Adjusted data
        for i, adj_method in enumerate(adjusted_methods, start=1):
            column_name = method + adj_method  # example: 'Median_LR'
            plt.plot(datasets[i]['beta'], datasets[i][column_name], linestyle=linestyles[i], marker='o', label=f'{method} {labels[i]}')

        plt.title(f'{method} Method Comparison')
        plt.xlabel('Beta')
        plt.ylabel('Power')
        plt.legend()
        plt.grid()
        plt.show()
