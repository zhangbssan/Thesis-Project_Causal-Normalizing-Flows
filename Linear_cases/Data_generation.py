import numpy as np
import Gaussian_generation
# generate one data point using inverted form 
# input: causal graph with parameter and noise. Input form: array,matrix,list
# output: one data point.Output form: if there is no matrix input, then the output is array. Or it is a matrix
def linear(G,input):
        output = np.matmul(G,input) 
        return output

