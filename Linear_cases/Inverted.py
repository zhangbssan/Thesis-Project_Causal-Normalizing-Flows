import numpy as np

def Linear(G,input):
        output = np.matmul(G,input) 
        return output
#X_f = linear((I-G).I,U).tolist()

# get one data couterfactual point when do intervention on i feature
def Get_intervention(G,I,X_f,X_new,i):
        U_obs   = Linear((I-G),X_f) # get the exo
        X_inter = [X_new if j == i else X_f[j] for j in range(0,len(X_f))] 
        U_new   = Linear((I-G),X_inter)
        U_inter = [U_new[j] if j == i else U_obs[j] for j in range(0,len(U_obs))]
        X_cf    = Linear(np.linalg.inv((I-G)),U_inter)
        return X_cf
# input one data couterfactual point matrix when do intervention on every feature
def Intervention_array_one(G,I,X_f,X_new):
   X_cf_matrix = np.transpose(np.array([Get_intervention(G,I,X_f,X_new,i) for  i in range(len(G))] ))
   return X_cf_matrix
# get the   
def Intervention_array_set(G,I,X_f,X_new):
      pass

# def Error_matrix_seq(*args,X,I,Error_matrix(G1,)):
#     def wrapper_do_seq():
#        for arg in args:
#             Error_matrix(G1,arg,X,I)
#         #error_mean_seq = [Error_matrix(G1,arg,X,I)[0]for arg in args]
#         #error_std_seq  = [Error_matrix(G1,arg,X,I)[1]for arg in args]
#        return wrapper_do_seq


def Error_matrix(G1,G2,X,I):
    counterfactual_base = [Intervention_array_one(G1,I,X[i],0.3)for i in range(len(X))]
    counterfactual_new =  [Intervention_array_one(G2,I,X[i],0.3)for i in range(len(X))]
    error_different = np.array(counterfactual_base) - np.array(counterfactual_new)
    error_square = (error_different**2).sum(axis=1)
    error = np.sqrt(error_square)
    ## 
    error_mean = np.mean(error,axis =0)
    error_std = np.std(error,axis =0)
    # the error between observed data
    # nonzero_elements = np.transpose(np.nonzero(error[0].round(6)))
    # error_mean = np.zeros_like(error[0])
    # error_std  = np.zeros_like(error[0])
    # for i in nonzero_elements:
    #     error_mean[tuple(i)] = np.mean([error[j][tuple(i)]for j in range(len(X))])
    #     error_std[tuple(i)] = np.std([error[j][tuple(i)]for j in range(len(X))])
    return error_mean,error_std

def Error_matrix_seq(G1,*args,X,I):
    error_mean_seq = [Error_matrix(G1,arg,X,I)[0]for arg in args]
    error_std_seq  = [Error_matrix(G1,arg,X,I)[1]for arg in args]
    return error_mean_seq,error_std_seq

# without intervention
def Error_obs(G,X,I):
    error_obs = np.array([Linear((I-G),X[i])for i in range(len(X))])
    error_square = (error_obs**2).sum(axis=1)
    error = np.sqrt(error_square)
    ## 
    error_mean = np.mean(error,axis =0)
    error_std = np.std(error,axis =0)
    # the error between observed data
    # nonzero_elements = np.transpose(np.nonzero(error[0].round(6)))
    # error_mean = np.zeros_like(error[0])
    # error_std  = np.zeros_like(error[0])
    # for i in nonzero_elements:
    #     error_mean[tuple(i)] = np.mean([error[j][tuple(i)]for j in range(len(X))])
    #     error_std[tuple(i)] = np.std([error[j][tuple(i)]for j in range(len(X))])
    return error_mean,error_std


def Error_matrix_per(G1,G2,X,I):
    counterfactual_base = [Intervention_array_one(G1,I,X[i],0.3)for i in range(len(X))]
    counterfactual_new =  [Intervention_array_one(G2,I,X[i],0.3)for i in range(len(X))]
    error_different = np.array(counterfactual_base) - np.array(counterfactual_new)
    error_square = error_different**2
    error = np.sqrt(error_square)
    ## 
    # error_mean = np.mean(error,axis =0)
    # error_std = np.std(error,axis =0)
    # the error between observed data
    nonzero_elements = np.transpose(np.nonzero(error[0].round(6)))
    error_mean_per = np.zeros_like(error[0])
    error_std_per  = np.zeros_like(error[0])
    for i in nonzero_elements:
        error_mean_per[tuple(i)] = np.mean([error[j][tuple(i)]for j in range(len(X))])
        error_std_per[tuple(i)] = np.std([error[j][tuple(i)]for j in range(len(X))])
    return error_mean_per,error_std_per

def Error_matrix_seq_per(G1,*args,X,I):
    error_mean_seq = [Error_matrix_per(G1,arg,X,I)[0]for arg in args]
    error_std_seq  = [Error_matrix_per(G1,arg,X,I)[1]for arg in args]
    return error_mean_seq,error_std_seq

            
      
                         
