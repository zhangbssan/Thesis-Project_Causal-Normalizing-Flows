import pandas as pd
import numpy as np

def Excel_output(error_matrix_mean,error_matrix_std):
    df_mean = [pd.DataFrame(np.array(i).round(6),
                  index=['x_1_cf_error', 'x_2_cf_error', 'x_3_cf_error','x_4_cf_error'], 
                  columns=['do(x1=0.3)', 'do(x2=0.3)', 'do(x3=0.3)','do(x4=0.3)'])for i in error_matrix_mean]
    df_std = [pd.DataFrame(np.array(i).round(6),
                  index=['x_1_cf_error', 'x_2_cf_error', 'x_3_cf_error','x_4_cf_error'], 
                  columns=['do(x1=0.3)', 'do(x2=0.3)', 'do(x3=0.3)','do(x4=0.3)'])for i in error_matrix_std]

    # df_mean = [pd.DataFrame(np.array(i).round(6),
    #               columns=['cf_error'], 
    #               index=['do(x1=0.3)', 'do(x2=0.3)', 'do(x3=0.3)','do(x4=0.3)'])for i in error_matrix_mean]
    # df_std = [pd.DataFrame(np.array(i).round(6),
    #               columns=['cf_error'], 
    #               index=['do(x1=0.3)', 'do(x2=0.3)', 'do(x3=0.3)','do(x4=0.3)'])for i in error_matrix_std]
    return df_mean,df_std