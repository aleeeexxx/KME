import os
import numpy as np
import pandas as pd
import scipy.linalg
# import matplotlib.pyplot as plt
# import scipy.linalg.hadamard as hadamard

def get_mc_traj(len):
    n_state = 9 # 1,2,3...9
    p = np.ndarray([n_state,n_state])
    p_gt = np.ndarray([n_state,n_state])
    for i in range(n_state):
        for j in range(n_state):
            if j == 0:
                p[i,j] =  np.exp(-(i+1))*np.exp(-(j+1)) + 2*(1-np.exp(-(i+1)))*np.exp(-2*(j+1))
                p_gt[i,j] =  p[i,j]
            else:
                p_gt[i,j] = np.exp(-(i+1))*np.exp(-(j+1)) + 2*(1-np.exp(-(i+1)))*np.exp(-2*(j+1)) #
                p[i,j] = p[i,j-1] + p_gt[i,j]
        p[i,:] = p[i,:] / p[i,n_state-1]
    p_gt = p_gt/p_gt.sum(axis=1)
    print("p_gt",p_gt)
    traj = np.ndarray([1,len])
    s = 0
    traj[0] = s
    counter = 0
    for i in range(len):
        s = get_next_state(s,p)
        traj[:,i] = s + 1
        if s is not 0:
            counter += 1
#         print(counter)
    return traj, p_gt
    
def get_next_state(i,p):
    prob = np.random.rand()
    for j in range(len(p[0])):
        if prob < p[i,j]:
            return j
    return j

def brownian_motion(len):
    dt = 0.01
    p = np.array([10.0,10.0])
    traj = np.ndarray([2,len])
    for i in range(len):
        p += np.sqrt(dt) * np.random.normal(loc=0.0,scale=1.0,size = 2)
        traj[:,i] = p
#     print(traj)
    return traj

def normalize(rrf):
    norm = np.linalg.norm(rrf)
    norm_rrf = rrf/norm
    return norm_rrf

def Get_p_(p_hat,r,c_rff_12,c_12):
    p_ = np.empty_like(p_hat)
    p_hat[0] = np.dot(np.dot(c_rff_12,p_hat[2]),c_rff_12)
    p_hat[2] = np.dot(np.dot(c_12,p_hat[2]),c_12)
    for i in range(len(p_hat)):
        u, s, vh = np.linalg.svd(p_hat[i], full_matrices=True)
        u_ = u[:,:r]
        s_ = np.diag(s[:r])

        vh_ = vh[:r,:]
        # print("s_: ",s_)
        p_[i] = np.dot(np.dot(u_,s_),vh_)
        # print("p_[1]", p_[1])
        # print("p_[2]", p_[2])
        # print("p_[3]", p_[3])

    return p_

def Get_P_hat(traj,G,n_feature, w_orf, w_sorf,step=1):
    avg = np.zeros(shape=(4,n_feature,n_feature))
    i = 0
    counter = 0
    # print(len(traj))
    while i+step < len(traj[0]):
        if i % 1000 == 0:
            print("current step",i)
        x = traj[:,i]
        y = traj[:,i+step]
        k_rff_1, k_rff_2, k_orf, k_sorf = Kernel(x,y,G, w_orf, w_sorf)
#         print("k_orf",k_orf)
        avg[0] += k_rff_1
        avg[1] += k_rff_2
        avg[2] += k_orf
        avg[3] += k_sorf
        
        i += step
        counter += 1
#         print("counter", counter)
    
    avg = avg/counter
    # print("p_hat",avg)
    return avg

def Kernel(x,y,G,w_orf,w_sorf):
    rff_1_x = RFF_1(x,w,n_raw,n_feature)
    rff_1_y = RFF_1(y,w,n_raw,n_feature)
    k_rff_1 = np.dot(rff_1_x.transpose(),rff_1_y)
    # print("rff_1_x", rff_1_x)
    # print("rff_1_y.shape", rff_1_y.shape)
    
    rff_2_x = RFF_2(x,G)
    rff_2_y = RFF_2(y,G)
    k_rff_2 = np.dot(rff_2_x.transpose(),rff_2_y)
    # print("rff_2_x", rff_2_x)
    # print("rff_2_y.shape", rff_2_y.shape)

    orf_x = ORF(x,w_orf)
    orf_y = ORF(y,w_orf)
    k_orf = np.dot(orf_x.transpose(),orf_y)
    # print("orf_x", orf_x)
    # print("orf_y.shape", orf_y.shape)

    sorf_x = SORF(x,w_sorf)
    sorf_y = SORF(y,w_sorf)
    k_sorf = np.dot(sorf_x.transpose(),sorf_y)
    # print("sorf_x", sorf_x)
    # print("sorf_y.shape", sorf_y.shape)

    # print("k_rff_1", k_rff_1)
    # print("k_rff_2", k_rff_2)
    # print("k_orf", k_orf)
    # print("k_sorf", k_sorf)
    return  k_rff_1,k_rff_2,k_orf, k_sorf  

# random Fourier feature 
def RFF_1(x,w,n_raw,n_feature):
    rff = np.zeros((n_raw,n_feature))
    rff[0,:]  = np.sqrt(2.0/n_feature)*(np.sin(w*x[0])) 
    
    # q, r = np.linalg.qr(rff_x)
    # print("q",q)
    # print("r",r)
    if n_raw == 2:
        rff[1,:] = np.sqrt(2.0/n_feature)*(np.cos(w*x[1]))  # D*1 
    return rff  #D*2

def RFF_2(x,G):
    rff = np.reshape(np.dot(G,x.transpose()),(1,-1))
    rff = normalize(rff)
    return rff # 2*D

def get_W(n_raw,n_feature):
    W = 2*np.pi*np.random.choice(2000,size=(n_raw,n_feature), replace=False)
    sum = 0

    for j in range(9):
        sum += np.sin(W*j)**2
    
    norm = np.sqrt(sum)
    
    c_12 = np.diag(1/norm.reshape(n_feature))
    # print("c_12",c_12)
    # print("W",W)
    return c_12, W

def get_w_orf(n_raw,n_feature,G):
    w_orf = np.zeros((n_feature,n_raw))
    c = np.zeros(n_feature)
    for i in range(n_feature//n_raw):
        q, r = np.linalg.qr(G[n_raw*i:n_raw*i+n_raw,:])
        s = np.random.chisquare(n_raw,size=n_raw)
        s_diag = np.diag(s)
        # print("s",s_diag)
        c[2*i:2*i+2] = 1/np.sqrt(s)
        w_orf[2*i:2*i+2] = np.dot(s_diag,q)
   
    c_diag = np.diag(c)
    return c_diag, w_orf

def ORF(x,w_orf):
    orf = np.reshape(np.dot(w_orf,x.transpose()),(1,-1))
#     print(orf)
    return  orf

def get_w_sorf(n_raw, n_feature):
    H = scipy.linalg.hadamard(n_raw)
    # print("H", H)
    w_sorf = np.zeros((n_feature,n_raw))
    for i in range(n_feature//n_raw):
        d1 = np.random.rand(n_raw)
        d2 = np.random.rand(n_raw)
        d3 = np.random.rand(n_raw)
        for element in d1:
            if element > 0.5:
                element = 1
            else:
                element = -1
        for element in d2:
            if element > 0.5:
                element = 1
            else:
                element = -1
        for element in d3:
            if element > 0.5:
                element = 1
            else:
                element = -1
        d1 = np.diag(d1)
        d2 = np.diag(d2)
        d3 = np.diag(d3)
        w_sorf[n_raw*i:n_raw*i+n_raw,:] = np.sqrt(2)*np.dot(np.dot(np.dot(np.dot(np.dot(H,d1),H),d2),H),d3)
   
    # sorf= normalize(sorf)
    return w_sorf
def SORF(x,w_sorf):
    sorf = np.reshape(np.dot(w_sorf,x),(1,-1))
    # sorf= normalize(sorf)
    return sorf

# traj = brownian_motion(10000)
traj, gt_p = get_mc_traj(5000)
print("random walk in [1, 2, 3... ,9] traj:",traj)
n_state = 9
r = 4 # rank
n_feature_list = [256] # n features
n_raw = 1 # n-dimensional raw data
step_list = [1]
x_list = [np.array([1]),np.array([2]),np.array([3])]
y_list = [np.array([1]),np.array([2]),np.array([3])]

for n_feature in n_feature_list:
    for step in step_list:
        print("n feature:",n_feature)
        print("step:",step)
        # w for rff_1
        w = np.random.normal(loc=0.0, scale=1.0, size=n_feature) # n * 1
        c_rff_12 , W = get_W(n_raw , n_feature)
        # gaussian for rff_2
        G = np.random.normal(loc=0.0, scale=1.0, size=(n_feature,n_raw))
        c_12 , w_orf = get_w_orf(n_raw,n_feature,G)
        w_sorf = get_w_sorf(n_raw,n_feature)
        p_hat = Get_P_hat(traj, G ,n_feature, w_orf, w_sorf,step)
        print("p_hat",p_hat) 
        p_ = Get_p_(p_hat,r,c_rff_12,c_12)

        # # print("p_hat:",p_hat)
        print("p_:",p_)
        p_est = np.zeros(shape=(4,n_state,n_state))
        for i in range(n_state):
            for j in range(n_state):
                x = np.array([i+1])
                y = np.array([j+1])

                rff_1_x = RFF_1(x,W ,n_raw, n_feature)
                rff_1_y = RFF_1(y,W ,n_raw, n_feature)
                rff_2_x = RFF_2(x, G)
                rff_2_y = RFF_2(y, G)

                orf_x  = ORF(x,w_orf)            
                orf_y = ORF(y,w_orf)                
                sorf_x = SORF(x,w_sorf)                
                sorf_y = SORF(y,w_sorf)            
                
                p_rff_1 = np.dot(np.dot(np.dot(np.dot(rff_1_x,c_rff_12), p_[0]),c_rff_12),rff_1_y.transpose())
                p_rff_2 = np.dot(np.dot(rff_2_x, p_[1]),rff_2_y.transpose())
                p_orf = np.dot(np.dot(np.dot(np.dot(orf_x,c_12), p_[0]),c_12),orf_y.transpose())
                p_sorf = np.dot(np.dot(sorf_x, p_[1]),sorf_y.transpose())
                
                # normalize
                

                p_est[0,i,j] = p_rff_1
                p_est[1,i,j] = p_rff_2
                p_est[2,i,j] = p_orf
                p_est[3,i,j] = p_sorf
        # normalize conditional prob that sum to 1  
        # print("sum",p_est[0].sum(axis=0))
        # sum = p_est[0].sum(axis=1)
        # p_est[0] = p_est[0]/sum
        # print(p_est[0])
        # p_est[0] = p_est[0]/p_est[0].sum(axis=1)
        # p_est[1] = p_est[1]/p_est[1].sum(axis=1)
        # print("p_est",p_est)
                

