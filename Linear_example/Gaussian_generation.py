import numpy as np
def Exo_matrix(mean,std_dev,size,len):
    exo = [np.random.normal(mean, std_dev, size) for i in range(len)]
    return exo
