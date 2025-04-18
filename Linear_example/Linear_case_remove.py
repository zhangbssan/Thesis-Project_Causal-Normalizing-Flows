import Inverted
import numpy as np
import pandas as pd


G_true = np.array([[0,0,0,0],
            [1,0,0,0],
            [2,3,0,0],
             [0,0,4,0]])

I = np.identity(4)
# remove the edge from x1 to x2
G_remove1  = np.matrix([[0,0,0,0],
            [0,0,0,0],
            [2,3,0,0],
             [0,0,4,0]])
# remove the edge from x1 to x3
G_remove2 = np.matrix([[0,0,0,0],
            [1,0,0,0],
            [0,3,0,0],
             [0,0,4,0]])
# remove the edge from x2 to x3
G_remove3 = np.matrix([[0,0,0,0],
            [1,0,0,0],
            [2,0,0,0],
             [0,0,4,0]])
# remove the edge from x3 to x4
G_remove4 = np.matrix([[0,0,0,0],
            [1,0,0,0],
            [2,3,0,0],
             [0,0,0,0]])
I = np.identity(4)
mean = 0.0
std_dev = 1.0
X_remove_error =np.zeros((4, 4))
X_remove_error2 =np.zeros((4, 4))
X_remove_error3 =np.zeros((4, 4))
X_remove_error4 =np.zeros((4, 4))

for i in range(0,100):
    U = [[np.random.normal(mean,std_dev)],
     [np.random.normal(mean,std_dev)],
     [np.random.normal(mean,std_dev)],
     [np.random.normal(mean,std_dev)]]
    # U = [[0.5],[0.5],[0.5],[0.5]]
    X_f = Inverted.Linear(np.linalg.inv(I-G_true),U).tolist() # data point observed
 
    X_remove_error= [[np.sum([X_remove_error[i][j] +
                               (Inverted.Get_intervention(G_true,I,X_f,0.3,j)[i,0]-Inverted.Get_intervention(G_remove1,I,X_f,0.3,j)[i,0])**2])
                               for j in range(len(U))] 
                               for i in range(4)]
    X_remove_error2= [[np.sum([X_remove_error2[i][j] + 
                               (Inverted.get_intervention(G_true,I,X_f,0.3,j)[i,0]-Inverted.get_intervention(G_remove2,I,X_f,0.3,j)[i,0])**2])
                               for  j in range(len(U))] 
                               for i in range(4)]
    X_remove_error3= [[np.sum([X_remove_error3[i][j] + 
                               (Inverted.get_intervention(G_true,I,X_f,0.3,j)[i,0]-Inverted.get_intervention(G_remove3,I,X_f,0.3,j)[i,0])**2])
                               for j in range(len(U))] 
                               for i in range(4)]
    X_remove_error4= [[np.sum([X_remove_error4[i][j] +
                               (Inverted.get_intervention(G_true,I,X_f,0.3,j)[i,0]-Inverted.get_intervention(G_remove4,I,X_f,0.3,j)[i,0])**2])
                               for j in range(len(U))] 
                               for i in range(4)]
     #list comprehension

df = pd.DataFrame(np.array(X_remove_error).round(6),
                  index=['x_1_cf_error', 'x_2_cf_error', 'x_3_cf_error','x_4_cf_error'], 
                  columns=['do(x1=0.3)', 'do(x2=0.3)', 'do(x3=0.3)','do(x4=0.3)'])

df2 = pd.DataFrame(np.array(X_remove_error2).round(6),
                  index=['x_1_cf_error', 'x_2_cf_error', 'x_3_cf_error','x_4_cf_error'], 
                  columns=['do(x1=0.3)', 'do(x2=0.3)', 'do(x3=0.3)','do(x4=0.3)'])
df3 = pd.DataFrame(np.array(X_remove_error3).round(6),
                  index=['x_1_cf_error', 'x_2_cf_error', 'x_3_cf_error','x_4_cf_error'], 
                  columns=['do(x1=0.3)', 'do(x2=0.3)', 'do(x3=0.3)','do(x4=0.3)'])
df4 = pd.DataFrame(np.array(X_remove_error4).round(6),
                  index=['x_1_cf_error', 'x_2_cf_error', 'x_3_cf_error','x_4_cf_error'], 
                  columns=['do(x1=0.3)', 'do(x2=0.3)', 'do(x3=0.3)','do(x4=0.3)'])
with pd.ExcelWriter('Remove.xlsx') as writer:
    df.to_excel(writer, sheet_name='x1tox2')
    df2.to_excel(writer, sheet_name='x1tox3')
    df3.to_excel(writer, sheet_name='x2tox3')
    df4.to_excel(writer, sheet_name='x3tox4')