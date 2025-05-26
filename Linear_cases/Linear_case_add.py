import Inverted
import numpy as np
import pandas as pd


G_true = np.matrix([[0,0,0,0],
            [1,0,0,0],
            [2,3,0,0],
             [0,0,4,0]])
## Case 1 (may some figure and table can be generated): Remove the edge from X2 to X3
I = np.identity(4)
# add the edge from x1 to x4
G_add1  = np.matrix([[0,0,0,0],
            [1,0,0,0],
            [2,3,0,0],
             [5,0,4,0]])
# add the edge from x2 to x3
G_add2 = np.matrix([[0,0,0,0],
            [1,0,0,0],
            [2,3,0,0],
             [0,5,4,0]])

I = np.identity(4)
mean = 0.0
std_dev = 1.0
X_add_error =np.zeros((4, 4))
X_add_error2 =np.zeros((4, 4))

for i in range(0,100):
    U = [[np.random.normal(mean,std_dev)],
     [np.random.normal(mean,std_dev)],
     [np.random.normal(mean,std_dev)],
     [np.random.normal(mean,std_dev)]]
    #U = [[0.5],[0.5],[0.5],[0.5]]
    X_f = Inverted.linear((I-G_true).I,U).tolist() # data point observed
 
    X_add_error= [[np.sum([X_add_error[i][j] +
                               (Inverted.get_intervention(G_true,I,X_f,0.3,j)[i,0]-Inverted.get_intervention(G_add1,I,X_f,0.3,j)[i,0])**2])
                               for j in range(len(U))] 
                               for i in range(4)]
    X_add_error2= [[np.sum([X_add_error2[i][j] + 
                               (Inverted.get_intervention(G_true,I,X_f,0.3,j)[i,0]-Inverted.get_intervention(G_add2,I,X_f,0.3,j)[i,0])**2])
                               for  j in range(len(U))] 
                               for i in range(4)]
    
     #list comprehension

df = pd.DataFrame(np.array(X_add_error).round(6),
                  index=['x_1_cf_error', 'x_2_cf_error', 'x_3_cf_error','x_4_cf_error'], 
                  columns=['do(x1=0.3)', 'do(x2=0.3)', 'do(x3=0.3)','do(x4=0.3)'])

df2 = pd.DataFrame(np.array(X_add_error2).round(6),
                  index=['x_1_cf_error', 'x_2_cf_error', 'x_3_cf_error','x_4_cf_error'], 
                  columns=['do(x1=0.3)', 'do(x2=0.3)', 'do(x3=0.3)','do(x4=0.3)'])

with pd.ExcelWriter('Add.xlsx') as writer:
    df.to_excel(writer, sheet_name='x1tox4')
    df2.to_excel(writer, sheet_name='x2tox4')
    