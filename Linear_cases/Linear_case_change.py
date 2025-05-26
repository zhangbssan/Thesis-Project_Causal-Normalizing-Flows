import Inverted
import numpy as np
import pandas as pd


G_true = np.matrix([[0,0,0,0],
            [1,0,0,0],
            [2,3,0,0],
             [0,0,4,0]])
I = np.identity(4)
# change the edge from x1 to x2
G_change1  = np.matrix([[0,1,0,0],
            [0,0,0,0],
            [2,3,0,0],
             [0,0,4,0]])
# change the edge from x2 to x3
G_change2 = np.matrix([[0,0,0,0],
            [1,0,3,0],
            [2,0,0,0],
             [0,0,4,0]])
# # change the edge from x1 to x3
# G_change3 = np.matrix([[0,0,2,0],
#             [1,0,0,0],
#             [0,3,0,0],
#              [0,0,4,0]])
# remove the edge from x3 to x4
G_change4 = np.matrix([[0,0,0,0],
            [1,0,0,0],
            [2,3,0,4],
             [0,0,0,0]])
I = np.identity(4)
mean = 0.0
std_dev = 1.0
X_change_error =np.zeros((4, 4))
X_change_error2 =np.zeros((4, 4))
X_change_error3 =np.zeros((4, 4))
X_change_error4 =np.zeros((4, 4))

for i in range(0,100):
    U = [[np.random.normal(mean,std_dev)],
     [np.random.normal(mean,std_dev)],
     [np.random.normal(mean,std_dev)],
     [np.random.normal(mean,std_dev)]]
    #U = [[0.5],[0.5],[0.5],[0.5]]
    X_f = Inverted.linear((I-G_true).I,U).tolist() # data point observed
 
    X_change_error= [[np.sum([X_change_error[i][j] +
                               (Inverted.get_intervention(G_true,I,X_f,0.3,j)[i,0]-Inverted.get_intervention(G_change1,I,X_f,0.3,j)[i,0])**2])
                               for j in range(len(U))] 
                               for i in range(4)]
    X_change_error2= [[np.sum([X_change_error2[i][j] + 
                               (Inverted.get_intervention(G_true,I,X_f,0.3,j)[i,0]-Inverted.get_intervention(G_change2,I,X_f,0.3,j)[i,0])**2])
                               for  j in range(len(U))] 
                               for i in range(4)]
    # X_change_error3= [[np.sum([X_change_error3[i][j] + 
    #                            (Inverted.get_intervention(G_true,I,X_f,0.3,j)[i,0]-Inverted.get_intervention(G_change3,I,X_f,0.3,j)[i,0])**2])
    #                            for j in range(len(U))] 
    #                            for i in range(4)]
    X_change_error4= [[np.sum([X_change_error4[i][j] +
                               (Inverted.get_intervention(G_true,I,X_f,0.3,j)[i,0]-Inverted.get_intervention(G_change4,I,X_f,0.3,j)[i,0])**2])
                               for j in range(len(U))] 
                               for i in range(4)]
     #list comprehension

df = pd.DataFrame(np.array(X_change_error).round(6),
                  index=['x_1_cf_error', 'x_2_cf_error', 'x_3_cf_error','x_4_cf_error'], 
                  columns=['do(x1=0.3)', 'do(x2=0.3)', 'do(x3=0.3)','do(x4=0.3)'])

df2 = pd.DataFrame(np.array(X_change_error2).round(6),
                  index=['x_1_cf_error', 'x_2_cf_error', 'x_3_cf_error','x_4_cf_error'], 
                  columns=['do(x1=0.3)', 'do(x2=0.3)', 'do(x3=0.3)','do(x4=0.3)'])
# df3 = pd.DataFrame(np.array(X_change_error3).round(6),
#                   index=['x_1_cf_error', 'x_2_cf_error', 'x_3_cf_error','x_4_cf_error'], 
#                   columns=['do(x1=0.3)', 'do(x2=0.3)', 'do(x3=0.3)','do(x4=0.3)'])
df4 = pd.DataFrame(np.array(X_change_error4).round(6),
                  index=['x_1_cf_error', 'x_2_cf_error', 'x_3_cf_error','x_4_cf_error'], 
                  columns=['do(x1=0.3)', 'do(x2=0.3)', 'do(x3=0.3)','do(x4=0.3)'])
with pd.ExcelWriter('Change.xlsx') as writer:
    df.to_excel(writer, sheet_name='x1tox2')
    df2.to_excel(writer, sheet_name='x2tox3')
    # df3.to_excel(writer, sheet_name='x1tox3')
    df4.to_excel(writer, sheet_name='x3tox4')