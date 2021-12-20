import os
from types import FunctionType
import numpy as np
import pandas as pd
import scipy.linalg
from scipy.optimize import LinearConstraint
from scipy.optimize import least_squares
# import matplotlib.pyplot as plt
# import scipy.linalg.hadamard as hadamard

def get_mc_traj(n_traj=3,n_state=3,len=10):
    p = np.zeros([n_traj,n_state,n_state])
    p_gt = np.zeros([n_traj,n_state,n_state])
    
    for k in range(n_traj):
        for i in range(n_state):
            for j in range(n_state):
                temp = np.random.rand()
                if temp > 0.5:
                    p_gt[k,i,j] = 1
                else:
                    p_gt[k,i,j] = 0
                if j == 0:
                    p[k,i,j] = p_gt[k,i,j]
                else:
                    p[k,i,j] = p[k,i,j-1] + p_gt[k,i,j]
            # print(" p[k,i,:]", p[k,i,:])   
            # print("p[k,i,n_state-1]", p[k,i,n_state-1])
            # if p[k,i,n_state-1] != 0:
            p[k,i,:] = p[k,i,:] / p[k,i,n_state-1]
            # print(" p[k,i,:]", p[k,i,:])   
        # print("k",k) 
        # print("p_gt",p_gt[k])   

        p_gt[k] = (p_gt[k].T/p_gt[k].sum(axis=1)).T
        p_gt[k] = np.where(np.isnan(p_gt[k]), 0, p_gt[k])
        # print("p_gt",p_gt[k])
      
    # initial state
    traj = np.zeros([n_traj,len])
    for k in range(n_traj):
        s = np.random.randint(n_state)
        traj[k,0] = s + 1
        for i in range(1,len):      
            # print("p[k]", p[k])
            s = get_next_state(s,p[k])
            traj[k,i] = s + 1
            # if s != 0:
            #     counter += 1
    #         print(counter)
    return traj, p_gt

def get_next_state(i,p):
    
    prob = np.random.rand()
    for j in range(len(p[0])):
        # print("p[i,j]",p[i,j])
        if prob < p[i,j]:
            return j
    return j

def get_p_bar(traj, n_state):
    total = len(traj)*(len(traj[0])-1)
    p_bar = np.zeros([n_state,n_state])
    # print("traj[0]",len(traj))
    # print("traj[1]",len(traj[1]))
    for k in range(len(traj)):
        for i in range(len(traj[1])-1):
            p_bar[int(traj[k,i])-1,int(traj[k,i+1])-1] += 1
    # print("p_bar",p_bar)
    p_bar = p_bar/total
    # print("total",total)
    # print("p_bar",p_bar)
    return p_bar

def get_alpha(p_bar,p_k):
    n = len(p_k)
    # print("n",n)
    x0 = np.array([1/n]*n)
    # print("x0",x0)
    res = least_squares(fun, x0, bounds=(0, 1),args=(p_bar, p_k), verbose=1)
    return res.x

def model(x, p_k):
    sum = 0
    for i in range(len(x)):
        sum += p_k[i]*x[i]
    return sum

def fun(x,p_bar,p_k):
    return sum((model(x, p_k) - p_bar)**2)



traj, p_k = get_mc_traj(5,5,100)
print("p_k",p_k)
# print("traj",traj)

p_bar = get_p_bar(traj, 5)
print("p_bar",p_bar)

alpha = get_alpha(p_bar,p_k)
print("alpha", alpha)