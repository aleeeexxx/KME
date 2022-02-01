import os
from types import FunctionType
import numpy as np
import pandas as pd
import scipy.linalg
from scipy.optimize import LinearConstraint
from scipy.optimize import least_squares
# import matplotlib.pyplot as plt
# import scipy.linalg.hadamard as hadamard

def get_gt(n_traj=10,n_state=3,len=10):
    p_gt = np.random.rand(n_state,n_state)       
    p_gt = (p_gt.T/p_gt.sum(axis=1)).T
    p_gt = np.where(np.isnan(p_gt), 0, p_gt)

    p = np.zeros([n_state,n_state])
    for i in range(n_state):
        for j in range(n_state):
            if j == 0:
                p[i,j] = p_gt[i,j]
            else:
                p[i,j] = p[i,j-1] + p_gt[i,j]

    # initial state
    traj = np.zeros([n_traj,len])
    for k in range(n_traj):
        s = np.random.randint(n_state)
        traj[k,0] = s + 1
        for i in range(1,len):      
            # print("p[k]", p[k])
            s = get_next_state(s,p)
            traj[k,i] = s + 1
            # if s != 0:
            #     counter += 1
    #         print(counter)
    return traj , p_gt

def get_p_k(n_traj=3,n_state=3):
    p = np.zeros([n_traj,n_state,n_state])
    p_k = np.zeros([n_traj,n_state,n_state])
    
    for k in range(n_traj):
        for i in range(n_state):
            for j in range(n_state):
                temp = np.random.rand()
                if temp > 0.5:
                    p_k[k,i,j] = 1
                else:
                    p_k[k,i,j] = 0
                if j == 0:
                    p[k,i,j] = p_k[k,i,j]
                else:
                    p[k,i,j] = p[k,i,j-1] + p_k[k,i,j]
            p[k,i,:] = p[k,i,:] / p[k,i,n_state-1] 
        p_k[k] = (p_k[k].T/p_k[k].sum(axis=1)).T
        p_k[k] = np.where(np.isnan(p_k[k]), 0, p_k[k])
        # print("p_gt",p_gt[k])
    return  p_k

def get_next_state(i,p):
    prob = np.random.rand()
    for j in range(len(p)):
        # print("p[i,j]",p[i,j])
        if prob < p[i,j]:
            return j
    return j

def get_p_bar(traj, n_state):
    # total = (len(traj[0])-1)
    p_bar = np.zeros([n_state,n_state])
    # print("traj[0]",len(traj))
    # print("traj[1]",len(traj[1]))
    for k in range(len(traj)):
        for i in range(len(traj[1])-1):
            p_bar[int(traj[k,i])-1,int(traj[k,i+1])-1] += 1
    p_bar = (p_bar.T/p_bar.sum(axis=1)).T
    p_bar = np.where(np.isnan(p_bar), 0, p_bar)
    return p_bar

def get_alpha(p_bar,p_k):
    n = len(p_k)
    # print("n",n)
    x0 = np.array([1/n]*n)
    # print("x0",x0)
    res = least_squares(fun, x0, bounds=(0, 1),args=(p_bar, p_k), verbose=1)
    alpha = res.x/sum(res.x)
    return alpha

def model(x, p_k):
    sum = 0
    for i in range(len(x)):
        sum += p_k[i]*x[i]
    return sum

def fun(x,p_bar,p_k):
    return sum((model(x, p_k) - p_bar)**2)

def get_p_hat(alpha,p_k):
    sum = 0
    for i in range(len(alpha)):
        sum += p_k[i]*alpha[i]
    return sum

def get_error(p_hat,Frobenius_gt):
    delta = p_hat - p_gt
#     print("delta",delta)
    Frobenius_delta = np.linalg.norm(delta)
    
#     print("Frobenius_gt",Frobenius_gt)
    error = Frobenius_delta/Frobenius_gt
    
    return error

traj, p_gt = get_gt(n_traj=1000,n_state=5,len=3000)
print("p_gt",p_gt)
Frobenius_gt = np.linalg.norm(p_gt)
print("Frobenius_gt",Frobenius_gt)

k_list = np.linspace(100, 500, 5, dtype=int,endpoint=True)
print("k_list",k_list)
error_list = []

p_k = get_p_k(n_traj=10,n_state=5)
p_bar = get_p_bar(traj, n_state=5)
alpha = get_alpha(p_bar,p_k)
p_hat = get_p_hat(alpha,p_k)
error = get_error(p_hat,Frobenius_gt)
print("error",error)
error_list.append(error)
print(error_list)

p_k = get_p_k(n_traj=50,n_state=5)
p_bar = get_p_bar(traj, n_state=5)
alpha = get_alpha(p_bar,p_k)
p_hat = get_p_hat(alpha,p_k)
error = get_error(p_hat,Frobenius_gt)
print("error",error)
error_list.append(error)
print(error_list)

p_k = get_p_k(n_traj=100,n_state=5)
p_bar = get_p_bar(traj, n_state=5)
alpha = get_alpha(p_bar,p_k)
p_hat = get_p_hat(alpha,p_k)
error = get_error(p_hat,Frobenius_gt)
print("error",error)
error_list.append(error)
print(error_list)

p_k = get_p_k(n_traj=200,n_state=5)
p_bar = get_p_bar(traj, n_state=5)
alpha = get_alpha(p_bar,p_k)
p_hat = get_p_hat(alpha,p_k)
error = get_error(p_hat,Frobenius_gt)
print("error",error)
error_list.append(error)
print(error_list)

p_k = get_p_k(n_traj=400,n_state=5)
p_bar = get_p_bar(traj, n_state=5)
alpha = get_alpha(p_bar,p_k)
p_hat = get_p_hat(alpha,p_k)
error = get_error(p_hat,Frobenius_gt)
print("error",error)
error_list.append(error)
print(error_list)