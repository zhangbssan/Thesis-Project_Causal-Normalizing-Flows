import pytest
import Inverted
import Recursive
import numpy as np

## Remove some edge 
G = np.matrix([[0,0,0,0],
            [1,0,0,0],
            [2,3,0,0],
             [0,0,4,0]])
G_remove  = np.matrix([[0,0,0,0],
            [1,0,0,0],
            [0,3,0,0],
             [0,0,4,0]])
G_change = np.matrix([[0,0,0,0],
            [1,0,3,0],
            [2,0,0,0],
             [0,0,4,0]])

G_add  = np.matrix([[0,0,0,0],
            [1,0,0,0],
            [2,3,0,0],
             [0,5,4,0]])

I = np.identity(4)
mean = 0.0
std_dev = 1.0
#f'{np.mean(i):.6f}'
U = [[np.random.normal(mean,std_dev)],
     [np.random.normal(mean,std_dev)],
     [np.random.normal(mean,std_dev)],
     [np.random.normal(mean,std_dev)]]
# U = [[0.5],
#      [0.5],
#      [0.5],
#      [0.5]]

X_f = Inverted.Linear((I-G).I,U).tolist() # data point observed


                                                                 
def test_Linear_case_linear(): 
    scm_generation = Recursive.SCM(1,2,3,4,U[0][0],U[1][0],U[2][0],U[3][0])
    assert round(X_f[0][0],6)== round(scm_generation.x1_obs,6)
    assert round(X_f[1][0],6) == round(scm_generation.x2_obs,6)
    assert round(X_f[2][0],6) == round(scm_generation.x3_obs,6)
    assert round(X_f[3][0],6) == round(scm_generation.x4_obs,6)

def test_Linear_case_get_intervention():
    X_cf = Inverted.Get_intervention(G,I,X_f,0.3,1).tolist()
    scm_intervention = Recursive.SCM_INTERVENTION(1,2,3,4,U[0][0],U[1][0],U[2][0],U[3][0])
    assert round(X_cf[0][0],6) == round(scm_intervention.x1_obs,6)
    assert round(X_cf[1][0],6) == round(scm_intervention.x2_obs,6)
    assert round(X_cf[2][0],6) == round(scm_intervention.x3_obs,6)
    assert round(X_cf[3][0],6) == round(scm_intervention.x4_obs,6)

def test_Linear_case_get_intervention_remove():
    X_cf = Inverted.Get_intervention(G_remove,I,X_f,0.3,1).tolist()
    scm_generation = Recursive.SCM(1,2,3,4,U[0][0],U[1][0],U[2][0],U[3][0])
    scm_add_exo = Recursive.SCM_REMOVE_EXO(1,2,3,4,scm_generation)
    scm_add_intervention = Recursive.SCM_REMOVE_INTERVENTION(1,2,3,4,scm_add_exo)
    assert round(X_cf[0][0],6) == round(scm_add_intervention.x1_obs,6)
    assert round(X_cf[1][0],6) == round(scm_add_intervention.x2_obs,6)
    assert round(X_cf[2][0],6) == round(scm_add_intervention.x3_obs,6)
    assert round(X_cf[3][0],6) == round(scm_add_intervention.x4_obs,6)

def test_Linear_case_get_intervention_change():
    X_cf = Inverted.Get_intervention(G_change,I,X_f,0.3,1).tolist()
    scm_generation = Recursive.SCM(1,2,3,4,U[0][0],U[1][0],U[2][0],U[3][0])
    scm_change_exo = Recursive.SCM_CHANGE_EXO(1,2,3,4,scm_generation)
    scm_change_intervention = Recursive.SCM_CHANGE_INTERVENTION(1,2,3,4,scm_change_exo)
    assert round(X_cf[0][0],6) == round(scm_change_intervention.x1_obs,6)
    assert round(X_cf[1][0],6) == round(scm_change_intervention.x2_obs,6)
    assert round(X_cf[2][0],6) == round(scm_change_intervention.x3_obs,6)
    assert round(X_cf[3][0],6) == round(scm_change_intervention.x4_obs,6)

def test_Linear_case_get_intervention_add():
    X_cf = Inverted.Get_intervention(G_add,I,X_f,0.3,1).tolist()
    scm_generation = Recursive.SCM(1,2,3,4,U[0][0],U[1][0],U[2][0],U[3][0])
    scm_add_exo = Recursive.SCM_ADD_EXO(1,2,3,4,5,scm_generation)
    scm_add_intervention = Recursive.SCM_ADD_INTERVENTION(1,2,3,4,5,scm_add_exo)
    assert round(X_cf[0][0],6) == round(scm_add_intervention.x1_obs,6)
    assert round(X_cf[1][0],6) == round(scm_add_intervention.x2_obs,6)
    assert round(X_cf[2][0],6) == round(scm_add_intervention.x3_obs,6)
    assert round(X_cf[3][0],6) == round(scm_add_intervention.x4_obs,6)