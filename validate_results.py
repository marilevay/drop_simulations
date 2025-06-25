import numpy as np
import pandas as pd
import os

def mean_square_error(y_true, y_pred):
    """Calculate the Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def validate_results(df_python, df_julia):
    # find csv file in the directory with the name beginning with "simulation_results"
    # read the file into a DataFrame
    # calculate the mean squared error against each colunm in the dataframe
    mse_results = {}
    for column in df_python.columns:
        if column in df_julia.columns:
            mse = mean_square_error(df_python[column], df_julia[column])
            mse_results[column] = mse
        else:
            print(f"Column {column} not found in Julia results.")
    return mse_results
    # average all results to obtain a single value that represents the capability of the new julia model to work with the simulation

if __name__ == "__main__":
    def find_mode_convergence_files(sweep_dir="drop_simulations_sweep/Mode_Convergence_R=nothing_modes=40_Bo=0_Oh=0", python_dir="python_results"):
        julia_files = []
        python_files = []
        for root, dirs, filenames in os.walk(sweep_dir):
            for fname in filenames:
                if "simulation_results" in fname and fname.endswith(".csv"):
                    julia_files.append(os.path.join(root, fname))
        for root, dirs, filenames in os.walk(python_dir):
            for fname in filenames:
                if "Bo=0_We=sweep" in fname and fname.endswith(".csv"):
                    python_files.append(os.path.join(root, fname))
        return julia_files, python_files
    
    julia_files, python_files = find_mode_convergence_files()
    print(f"Found {len(julia_files)} Julia files and {len(python_files)} Python files for comparison.")

    for py_file in python_files:
        df_python = pd.read_csv(py_file)
        for julia_file in julia_files:
            df_julia = pd.read_csv(julia_file, usecols=['R_in_CGS', 'V_in_CGS', 'Bo', 'We', 'Coefficient of Restitution', 'Contact Time', 'Maximum Radius'])
            print(f"\nComparing Python file '{py_file}' with Julia file '{julia_file}':")
            mse_results = validate_results(df_python, df_julia)
            average_mse = np.mean(list(mse_results.values()))
            print(f"Mean Squared Error for each column: {mse_results}")
            print(f"Average Mean Squared Error: {average_mse}")
    