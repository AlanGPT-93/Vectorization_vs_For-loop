import pandas as pd
import time
import numpy as np

def crate_df(num_rows, zero_percentage, num_columns):

    # Generate random float values between 0 and 1
    values = np.random.rand(num_rows, num_columns)

    # Set % of the values to 0
    mask = np.random.choice([0, 1], size=(num_rows, num_columns), p=[zero_percentage, 1 - zero_percentage])
    values = values * mask

    # Shuffle the values within each column
    np.random.shuffle(values)

    # Create a DataFrame from float values
    df = pd.DataFrame(values, columns = float_columns)

    # Create string columns 
    str_values = ["Acciones IPC", "Acciones y otros valores", "Prenda", "Inmuebles", "Dinero y valores" ,"Ent con Gar", "Der cobro", 
                  "Deuda Emisores", "Fideic", "Hipoteca", "Ap Federales", "Depositos", "Gar Expresa", "Fiduciarios", "AVAL"]
    str_ln = len(str_values)
    new_cols = np.where(df[float_columns] == 0, "", np.random.choice(str_values, size=(num_rows, num_columns), p = [1 / str_ln ] * str_ln ) )

    # Append string columns in the DataFrame
    df[string_columns] = new_cols

    # Create bigger float values
    bigger_values = (values * np.random.rand(num_rows, num_columns) ) * 1000

    # Append bigger columns in the DataFrame
    df[amount_columns] = bigger_values  

    return df


def get_sorted(row_values, core = "vectorization"):
    
    if core != "vectorization":
        x_sorted = row_values
        is_zero = np.where(x_sorted == 0)[0]
        x_sorted[is_zero] = np.inf

    else:
        x_sorted = np.where(row_values == 0.0, np.inf, row_values)
    
    return np.argsort(x_sorted)
    

def for_loop_execution(df):
  
    best_gar = [f"best_gar{i}" for i in range(1,6)] # new float columns
    cols =  np.array(float_columns)  # existing columns

    best_type = [f"best_type{i}" for i in range(1,6)] # new text columns
    type_cols = np.array(string_columns) # existing columns
    
    best_value = np.array([f"best_value{i}" for i in range(1,6)]) # new float columns
    value_cols = np.array(amount_columns) # existing columns

    for i_r in range(df.shape[0]):
        xs = get_sorted(row_values = df.loc[i_r, cols], core="for") # Sorted values with 0s in the last positions

        # BEST GAR
        df.loc[i_r, best_gar] = list(df.loc[i_r, cols[xs]])

        # BEST TYPE GAR
        df.loc[i_r, best_type] = list(df.loc[i_r, type_cols[xs]])

        # BEST TYPE GAR
        df.loc[i_r, best_value] = list(df.loc[i_r, value_cols[xs]])

        aux = np.where(df.loc[i_r, best_type] != "AVAL")[0]
        df.loc[i_r, "covered_amount"] = df.loc[i_r, best_value[aux]].sum()

    return df


def vectorization_execution(df):

    best_gar = [f"best_gar{i}" for i in range(1,6)] # new float columns
    cols =  float_columns  # existing columns

    best_type = [f"best_type{i}" for i in range(1,6)] # new text columns
    type_cols = string_columns # existing columns
    
    best_value = [f"best_value{i}" for i in range(1,6)] # new float columns
    value_cols = amount_columns # existing columns
    
    xs = get_sorted(row_values = df[cols].values ) # Sorted values with 0s in the last positions
    
    # BEST GAR
    df[best_gar] = np.take_along_axis(df[cols].values, xs, axis=1)
    
    # BEST TYPE GAR
    df[best_type] = np.take_along_axis(df[type_cols].values, xs, axis=1)

    # BEST VALUE
    df[best_value] = np.take_along_axis(df[value_cols].values, xs, axis=1)
    
    # COVERED AMOUNT FOR NON FINANCIAL GAR
    rc = np.where(df[best_type] == "AVAL", 0, df[best_value])
    df["covered_amount"] = rc.sum(axis=1)

    return df


if __name__ == "__main__":
    
    # Define the dimensions of the DataFrame
    num_rows = 40000
    num_columns = 5
    zero_percentage = 0.8

    # Define the columns' names of the DataFrame
    float_columns = [f'float_column{i}' for i in range(1, 6)]
    string_columns = [f'string_columns{i}' for i in range(1, 6)]
    amount_columns = [f'amount_columns{i}' for i in range(1, 6) ]
    
    # Create DataFrame
    df = crate_df(num_rows, zero_percentage, num_columns)

    # Doing process with vectorization
    start_time = time.time()
    df_vectorization = vectorization_execution(df.copy())
    print("\n Vectorization: --- %s seconds ---" % (time.time() - start_time))

    # Doing process with for loop
    start_time = time.time()
    df_for = for_loop_execution(df.copy())
    print("\n For Loop: --- %s seconds ---" % (time.time() - start_time))
