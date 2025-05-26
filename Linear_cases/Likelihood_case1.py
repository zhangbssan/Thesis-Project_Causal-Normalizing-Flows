import Inverted
import numpy as np
import Gaussian_generation
import Structure
import pandas as pd
import Output
import matplotlib.pyplot as plt
# Data generation 
G_true = np.array([[0,0,0,0],
            [1,0,0,0],
            [2,3,0,0],
             [0,0,4,0]])
I = np.identity(4)
mean = 0.0
std_dev = 1.0
U = Gaussian_generation.Exo_matrix(mean,std_dev,4,10000)
X_f = np.array([Inverted.Linear(np.linalg.inv((I-G_true)),U[i]) for i in range(10000)]).reshape(10000,len(I))
X_f
# plt.plot(np.array(U)[:,1])
# plt.show()
# plt.plot(X_f[:,1])
# plt.show()
# plt.plot(X_f[:,2])
# plt.show()

cols = 4
rows = 4

## on the estimated causal graph
## optimal solution
G_parameter = Structure.Likelihood_linear(cols,rows,(1,0),(2,0),(2,1),(3,2),la = 0.001,X=X_f,i =5)

df = pd.DataFrame(np.array(G_parameter).round(6),
                  index=['x_1', 'x_2', 'x_3','x_4'], 
                  columns=['x_1', 'x_2', 'x_3','x_4'])

# Remove
## remove x1 to x2
G_parameter1 = Structure.Likelihood_linear(cols,rows,(2,0),(2,1),(3,2),la = 0.001,X=X_f,i =5)

df1 = pd.DataFrame(np.array(G_parameter1).round(6),
                  index=['x_1', 'x_2', 'x_3','x_4'], 
                  columns=['x_1', 'x_2', 'x_3','x_4'])

## remove x1 to x3
G_parameter2 = Structure.Likelihood_linear(cols,rows,(1,0),(2,1),(3,2),la = 0.001,X=X_f,i =5)

df2 = pd.DataFrame(np.array(G_parameter2).round(6),
                  index=['x_1', 'x_2', 'x_3','x_4'], 
                  columns=['x_1', 'x_2', 'x_3','x_4'])
## remove x2 to x3
G_parameter3 = Structure.Likelihood_linear(cols,rows,(1,0),(2,0),(3,2),la = 0.001,X=X_f,i =5)

df3 = pd.DataFrame(np.array(G_parameter3).round(6),
                  index=['x_1', 'x_2', 'x_3','x_4'], 
                  columns=['x_1', 'x_2', 'x_3','x_4'])
## remove x3 to x4
G_parameter4 = Structure.Likelihood_linear(cols,rows,(1,0),(2,0),(2,1),la = 0.001,X=X_f,i =5)

df4 = pd.DataFrame(np.array(G_parameter4).round(6),
                  index=['x_1', 'x_2', 'x_3','x_4'], 
                  columns=['x_1', 'x_2', 'x_3','x_4'])
#G_parameter6 = Structure.Likelihood_linear(cols,rows,(1,0),(2,1),la = 0.001,X=X_f,i =5)

with pd.ExcelWriter('Remove_parameter.xlsx') as writer:
    df.to_excel(writer, sheet_name='Original')
    df1.to_excel(writer, sheet_name='x1tox2')
    df2.to_excel(writer, sheet_name='x1tox3')
    df3.to_excel(writer, sheet_name='x2tox3')
    df4.to_excel(writer, sheet_name='x3tox4')

# Add
## add x1 to x4
G_parameter5 = Structure.Likelihood_linear(cols,rows,(1,0),(2,0),(2,1),(3,2),(3,0),la = 0.001,X=X_f,i =5)
df5 = pd.DataFrame(np.array(G_parameter5).round(6),
                  index=['x_1', 'x_2', 'x_3','x_4'], 
                  columns=['x_1', 'x_2', 'x_3','x_4'])
## add x2 to x4
G_parameter6 = Structure.Likelihood_linear(cols,rows,(1,0),(2,0),(2,1),(3,2),(3,1),la = 0.001,X=X_f,i =5)
df6 = pd.DataFrame(np.array(G_parameter6).round(6),
                  index=['x_1', 'x_2', 'x_3','x_4'], 
                  columns=['x_1', 'x_2', 'x_3','x_4'])
with pd.ExcelWriter('Add_parameter.xlsx') as writer:
    df.to_excel(writer, sheet_name='Original')
    df5.to_excel(writer, sheet_name='x1tox4')
    df6.to_excel(writer, sheet_name='x2tox4')

# Change
## the direction between x1 to x2
G_parameter7 = Structure.Likelihood_linear(cols,rows,(0,1),(2,0),(2,1),(3,2),la = 0.001,X=X_f,i =5)
df7 = pd.DataFrame(np.array(G_parameter7).round(6),
                  index=['x_1', 'x_2', 'x_3','x_4'], 
                  columns=['x_1', 'x_2', 'x_3','x_4'])
## the direction between x2 to x3
G_parameter8 = Structure.Likelihood_linear(cols,rows,(1,0),(2,0),(1,2),(3,2),la = 0.001,X=X_f,i =5)
df8 = pd.DataFrame(np.array(G_parameter8).round(6),
                  index=['x_1', 'x_2', 'x_3','x_4'], 
                  columns=['x_1', 'x_2', 'x_3','x_4'])
## the direction between x3 to x4
G_parameter9 = Structure.Likelihood_linear(cols,rows,(1,0),(2,0),(2,1),(2,3),la = 0.001,X=X_f,i =5)
df9 = pd.DataFrame(np.array(G_parameter9).round(6),
                  index=['x_1', 'x_2', 'x_3','x_4'], 
                  columns=['x_1', 'x_2', 'x_3','x_4'])
with pd.ExcelWriter('Change_parameter.xlsx') as writer:
    df.to_excel(writer, sheet_name='Original')
    df7.to_excel(writer, sheet_name='x2tox1')
    df8.to_excel(writer, sheet_name='x3tox2')
    df9.to_excel(writer, sheet_name='x4tox3')
    
# print(G_parameter)
# print(G_parameter1)
# print(G_parameter2)
# print(G_parameter3)
# print(G_parameter4)
# print(G_parameter5)
# print(G_parameter6)
# print(G_parameter7)

# get the couterfactual error with G_True

# Remove

error_matrix_mean,error_matrix_std = Inverted.Error_matrix_seq(G_true,G_parameter1,G_parameter2,G_parameter3,G_parameter4,X=X_f,I=I)
data0 = np.array([error_matrix_mean[0],error_matrix_std[0]]).round(10)
df0 = pd.DataFrame(data0,columns=['do(x1=0.3)', 'do(x2=0.3)', 'do(x3=0.3)','do(x4=0.3)'], 
                  index=['cf_error','cf_error_std'])

data1 = np.array([error_matrix_mean[1],error_matrix_std[1]]).round(10)
df1 = pd.DataFrame(data1,columns=['do(x1=0.3)', 'do(x2=0.3)', 'do(x3=0.3)','do(x4=0.3)'], 
                  index=['cf_error','cf_error_std'])
data2 = np.array([error_matrix_mean[2],error_matrix_std[2]]).round(10)
df2 = pd.DataFrame(data2,columns=['do(x1=0.3)', 'do(x2=0.3)', 'do(x3=0.3)','do(x4=0.3)'], 
                  index=['cf_error','cf_error_std'])
data3 = np.array([error_matrix_mean[3],error_matrix_std[3]]).round(10)
df3 = pd.DataFrame(data3,columns=['do(x1=0.3)', 'do(x2=0.3)', 'do(x3=0.3)','do(x4=0.3)'], 
                  index=['cf_error','cf_error_std'])

with pd.ExcelWriter('Remove_likelihood_true.xlsx') as writer:
    df0.to_excel(writer, sheet_name='x1tox2')
    df1.to_excel(writer, sheet_name='x1tox3')
    df2.to_excel(writer, sheet_name='x2tox3')
    df3.to_excel(writer, sheet_name='x3tox4')



# Add
error_matrix_mean,error_matrix_std = Inverted.Error_matrix_seq(G_true,G_parameter5,G_parameter6,X=X_f,I=I)
data4 = np.array([error_matrix_mean[0],error_matrix_std[0]]).round(10)
df4 = pd.DataFrame(data4,columns=['do(x1=0.3)', 'do(x2=0.3)', 'do(x3=0.3)','do(x4=0.3)'], 
                  index=['cf_error','cf_error_std'])
data5 = np.array([error_matrix_mean[1],error_matrix_std[1]]).round(10)
df5 = pd.DataFrame(data5,columns=['do(x1=0.3)', 'do(x2=0.3)', 'do(x3=0.3)','do(x4=0.3)'], 
                  index=['cf_error','cf_error_std'])
# df_mean,df_std = Output.Excel_output(error_matrix_mean,error_matrix_std)
with pd.ExcelWriter('Add_likelihood_true.xlsx') as writer:
    df4.to_excel(writer, sheet_name='x1tox4')
    df5.to_excel(writer, sheet_name='x2tox4')


# with pd.ExcelWriter('Add_likelihood_std.xlsx') as writer:
#     df_std[0].to_excel(writer, sheet_name='x1tox4') 
#     df_std[1].to_excel(writer, sheet_name='x2tox4')

# Change


# error_matrix_mean,error_matrix_std = Inverted.Error_matrix_seq(G_true,G_parameter7,G_parameter8,G_parameter9,X=X_f,I=I)
# data6 = np.array([error_matrix_mean[0],error_matrix_std[0]]).round(10)
# df6= pd.DataFrame(data6,columns=['do(x1=0.3)', 'do(x2=0.3)', 'do(x3=0.3)','do(x4=0.3)'], 
#                   index=['cf_error','cf_error_std'])
# data7 = np.array([error_matrix_mean[1],error_matrix_std[1]]).round(10)
# df7 = pd.DataFrame(data7)