import numpy as np
import random

class SCM:
    def __init__(self,a,b,c,d,u1,u2,u3,u4):
        self.x1_obs = self.x1(u1)
        self.x2_obs = self.x2(a,u2)
        self.x3_obs = self.x3(b,c,u3)
        self.x4_obs = self.x4(d,u4)
        self.u1 = u1
        self.u2 = u2
        self.u3 = u3
        self.u4 = u4
        
           
    def x1(self,u1):
        x1 = u1
        return x1
    
    def x2(self,a,u2):
        x2 = a*self.x1_obs +u2
        return x2
    
    def x3(self,b,c,u3):
        x3 = b*self.x1_obs+c*self.x2_obs+u3
        return x3
    
    
    def x4(self,d,u4):
        x4 = d*self.x3_obs+u4
        return x4
    
class SCM_INTERVENTION(SCM):
    def __init__(self,a,b,c,d,u1,u2,u3,u4):
        #super().__init__(a,b,c,d,u1,u2,u3,u4)
        self.x1_obs = super().x1(u1)
        self.x2_obs = 0.3
        self.x3_obs = super().x3(b,c,u3)
        self.x4_obs = super().x4(d,u4)

class SCM_REMOVE_EXO():
    def __init__(self,a,b,c,d,scm_true_obs):
        self.x1_obs = scm_true_obs.x1_obs
        self.x2_obs = scm_true_obs.x2_obs
        self.x3_obs = scm_true_obs.x3_obs
        self.x4_obs = scm_true_obs.x4_obs
        self.u1_obs     = self.u1()
        self.u2_obs     = self.u2(a)
        self.u3_obs     = self.u3(b,c)
        self.u4_obs     = self.u4(d)
    
    def u1(self):
        u1 =  self.x1_obs
        return u1
  
    def u2(self,a):
        u2 = self.x2_obs - a*self.x1_obs 
        return u2
 
    def u3(self,b,c):
        u3 = self.x3_obs - b*self.x1_obs - c*self.x2_obs
        return u3
   
    def u4(self,d):
        u4 = self.x4_obs - d*self.x3_obs
        return u4

class SCM_REMOVE_INTERVENTION(SCM):
    def __init__(self,a,b,c,d,scm_remove_exo):
        self.u1     = scm_remove_exo.u1_obs
        self.u2     = scm_remove_exo.u2_obs
        self.u3     = scm_remove_exo.u3_obs
        self.u4     = scm_remove_exo.u4_obs
        self.x1_obs = super().x1(self.u1)
        self.x2_obs = 0.3
        self.x3_obs = super().x3(b,c,self.u3)
        self.x4_obs = super().x4(d,self.u4)

class SCM_CHANGE_EXO():
    def __init__(self,a,b,c,d,scm_true_obs):
        self.x1_obs = scm_true_obs.x1_obs
        self.x2_obs = scm_true_obs.x2_obs
        self.x3_obs = scm_true_obs.x3_obs
        self.x4_obs = scm_true_obs.x4_obs
        self.u1_obs     = self.u1()
        self.u2_obs     = self.u2(a,c)
        self.u3_obs     = self.u3(b)
        self.u4_obs     = self.u4(d)
    
    def u1(self):
        u1 =  self.x1_obs
        return u1
  
    def u2(self,a,c):
        u2 = self.x2_obs - a*self.x1_obs - c*self.x3_obs
        return u2
 
    def u3(self,b):
        u3 = self.x3_obs - b*self.x1_obs 
        return u3
   
    def u4(self,d):
        u4 = self.x4_obs - d*self.x3_obs
        return u4

class SCM_CHANGE_INTERVENTION(SCM):
    def __init__(self,a,b,c,d,scm_change_exo):
        self.u1     = scm_change_exo.u1_obs
        self.u2     = scm_change_exo.u2_obs
        self.u3     = scm_change_exo.u3_obs
        self.u4     = scm_change_exo.u4_obs
        self.x1_obs = super().x1(self.u1)
        self.x2_obs = 0.3
        self.x3_obs = super().x3(b,0,self.u3)
        self.x4_obs = super().x4(d,self.u4)

class SCM_ADD_EXO():
    def __init__(self,a,b,c,d,e,scm_true_obs):
        self.x1_obs = scm_true_obs.x1_obs
        self.x2_obs = scm_true_obs.x2_obs
        self.x3_obs = scm_true_obs.x3_obs
        self.x4_obs = scm_true_obs.x4_obs
        self.u1_obs     = self.u1()
        self.u2_obs     = self.u2(a)
        self.u3_obs     = self.u3(b,c)
        self.u4_obs     = self.u4(d,e)
    
    def u1(self):
        u1 =  self.x1_obs
        return u1
  
    def u2(self,a):
        u2 = self.x2_obs - a*self.x1_obs 
        return u2
 
    def u3(self,b,c):
        u3 = self.x3_obs - b*self.x1_obs -c*self.x2_obs
        return u3
   
    def u4(self,d,e):
        u4 = self.x4_obs - d*self.x3_obs -e*self.x2_obs
        return u4
    
class SCM_ADD_INTERVENTION(SCM):
    def __init__(self,a,b,c,d,e,scm_add_exo):
        self.u1     = scm_add_exo.u1_obs
        self.u2     = scm_add_exo.u2_obs
        self.u3     = scm_add_exo.u3_obs
        self.u4     = scm_add_exo.u4_obs
        self.x1_obs = super().x1(self.u1)
        self.x2_obs = 0.3
        self.x3_obs = super().x3(b,c,self.u3)
        self.x4_obs = d* self.x3_obs + e* self.x2_obs + self.u4