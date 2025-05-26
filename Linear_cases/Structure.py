import numpy as np
def G_structure(cols,rows,*args):
   G = np.array([[0 for _ in range(cols)] for _ in range(rows)])
   for arg in args:
      G[arg] = 1
   return G
def Phi(G,X):
   P = np.array([[0.0 for _ in range(np.sum(G)) ] for _ in range(np.count_nonzero(np.any( G != 0, axis=1)))])
   k=0
   z=0
   for i in range(len(G)):
      for j in range(len(G)):
         #if G[i,j] !=0 and j <(i+1):
         if G[i,j] !=0:
            P[z,k] = X[j]
            k += 1 
      if np.any(G[i] !=0):
        z+=1   
   return P

   #Phi =
def Phi_weights_part1(phi,la):
  
# i to n phi.T*phi
  W1 = [np.matmul(np.array(phi[i]).T,np.array(phi[i])) for i in range(len(phi))]
# set a matrix which has the same size with W1[0]
  global phi_rows 
  phi_rows = len(W1[0])
  global phi_cols
  phi_cols = len(W1[0][0])
  W2 = np.zeros((phi_rows,phi_cols))
# sum for 1 to n 
  for i in range(len(phi)):
      W2 += W1[i]
  if np.linalg.det(W2) == 0:
      return np.linalg.inv((W2+la*np.identity(phi_rows)))
  else:
      return np.linalg.inv(W2)
  
def Phi_weights_part2(phi,X,G):
  non_zero_indices = np.nonzero(np.any( G != 0, axis=1))
  #W3 = [np.matmul(np.array(phi[i]).T,np.array([X[i][1],X[i][2],X[i][3]]))for i in range(len(Phi2))]
  
  W3 = [np.matmul(np.array(phi[i]).T,np.transpose([X[i][row] for index,row in enumerate(non_zero_indices)]) )for i in range(len(phi))]
 
  W4 = np.zeros((phi_cols,1))

  for i in range(len(phi)):
    W4 +=W3[i]
  
  return W4

def Phi_weights(phi,X,G,la):
   estimate_w= np.matmul(np.matrix(Phi_weights_part1(phi,la)),Phi_weights_part2(phi,X,G))
   return estimate_w


def G_parameters(G,phi,w,X,i):
   W11 = np.matmul(phi, w).reshape(len(X),len(phi[0])).tolist()
 
   # add zero elements according to the structure of G
   for row in W11:
    for i in range(len(G)):
       if np.sum(G[i]) ==0:
          row.insert(i,0.)
   W15 = np.matmul(np.matrix(X).T,np.matrix(X))
   W16 = np.matmul(np.array(W11).T,X)
   G_p = np.array(np.matmul(W16,W15.I)).round(i)
   return G_p

def Likelihood_linear(cols,rows,*args,la,X,i):
  g = G_structure(cols,rows,*args)
  #phi = [Phi(g,X[i])for i in range(len(X))]
  phi = np.array([Phi(g,X[i])for i in range(len(X))])
  #phi = Phi(g,X)
  estimate_weights = Phi_weights(phi,X,g,la)
  g_parameters = G_parameters(g,phi,estimate_weights,X,i)
  return g_parameters



