import pandas as pd
import polars as pl
import numpy as np 

def reduce_mem_usage(df, float16_as32=True):
    # memory_usage() returns memory usage of each column, sum for total usage, convert B->KB->MB
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:  # Iterate through each column name
        col_type = df[col].dtype  # Get column data type
        if col_type != object and str(col_type)!='category':  # Process numeric types only (not object or category)
            c_min,c_max = df[col].min(),df[col].max()  # Get min and max values of this column
            if str(col_type)[:3] == 'int':  # If it's an integer type (int8, int16, int32, or int64)
                # If values are within int8 range, convert to int8 (-128 to 127)
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                # If values are within int16 range, convert to int16 (-32,768 to 32,767)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                # If values are within int32 range, convert to int32 (-2,147,483,648 to 2,147,483,647)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                # If values are within int64 range, convert to int64 (-9,223,372,036,854,775,808 to 9,223,372,036,854,775,807)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:  # If it's a floating point type
                # If values are within float16 range, consider float32 for better precision if needed
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    if float16_as32:  # Use float32 if higher precision is needed
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float16)  
                # If values are within float32 range, convert to float32
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                # If values are within float64 range, convert to float64
                else:
                    df[col] = df[col].astype(np.float64)
    # Calculate memory usage after optimization
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    # Calculate percentage reduction compared to initial memory usage
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df