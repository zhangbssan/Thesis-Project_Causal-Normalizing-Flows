import Inverted
import numpy as np
import Gaussian_generation
import Structure
import pandas as pd
import Output
# Data generation
G_true = np.array([[0,0,0,0,0],
            [1,0,0,0,0],
            [2,3,0,0,0],
             [0,0,4,0,0],
             [5,0,0,6,0]])
I = np.identity(5)
mean = 0.0
std_dev = 1.0
U = Gaussian_generation.Exo_matrix(mean,std_dev,5,500)
X_f = np.array([Inverted.Linear(np.linalg.inv((I-G_true)),U[i]) for i in range(500)]).reshape(500,len(I))
X_f


cols = 5
rows = 5

G_parameter = Structure.Likelihood_linear(cols,rows,(1,0),(2,0),(2,1),(3,2),(4,0),(4,3),la = 0.001,X=X_f,i =5)
G_parameter1 = Structure.Likelihood_linear(cols,rows,(1,0),(2,0),(2,1),(3,2),(4,0),la = 0.001,X=X_f,i =5)
G_parameter2 = Structure.Likelihood_linear(cols,rows,(1,0),(2,0),(2,1),(4,0),(4,3),la = 0.001,X=X_f,i =5)
G_parameter3 = Structure.Likelihood_linear(cols,rows,(1,0),(2,0),(2,1),(3,2),(4,0),(4,3),(4,2),la = 0.001,X=X_f,i =5)
print(G_parameter)
print(G_parameter1)
print(G_parameter2)