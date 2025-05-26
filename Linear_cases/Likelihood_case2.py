import Inverted
import numpy as np
import Gaussian_generation
import Structure
import pandas as pd

# Data generation
G_true = np.array([[0,0,0],
            [1,0,0],
            [2,3,0]])
I = np.identity(3)
mean = 0.0
std_dev = 1.0
U = Gaussian_generation.Exo_matrix(mean,std_dev,3,200)
X_f = np.array([Inverted.Linear(np.linalg.inv((I-G_true)),U[i]) for i in range(200)]).reshape(200,len(I))
X_f

# optimal solution
cols = 3
rows = 3

## on the estimated causal graph
## Remove some edge
G_parameter = Structure.Likelihood_linear(cols,rows,(1,0),(2,0),(2,1),la = 0.001,X=X_f,i =5)
G_parameter1 = Structure.Likelihood_linear(cols,rows,(2,0),(2,1),la = 0.001,X=X_f,i =5)
# G_parameter = Structure.Likelihood_linear(cols,rows,(1,0),(2,0),(2,1),la = 0.001,X=X_f,i =5)
#G_parameter1 = Structure.Likelihood_linear(cols,rows,(2,0),(2,1),la = 0.001,X=X_f,i =5)
G_parameter2 = Structure.Likelihood_linear(cols,rows,(1,0),(2,0),la = 0.001,X=X_f,i =5)
G_parameter3 = Structure.Likelihood_linear(cols,rows,(1,0),(2,1),la = 0.001,X=X_f,i =5)
G_parameter4 = Structure.Likelihood_linear(cols,rows,(1,0),la = 0.001,X=X_f,i =5)
#G_parameter4 = Structure.Likelihood_linear(cols,rows,(1,0),(2,1),(3,2),la = 0.001,X=X_f,i =5)
# get the couterfactual error






