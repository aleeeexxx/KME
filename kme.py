
import os
import numpy as np
import pandas as pd
import scipy.linalg
# import matplotlib.pyplot as plt
# import scipy.linalg.hadamard as hadamard

def get_mc_traj(len):
  n_state = 9 # 1,2,3...9
  p = np.ndarray([n_state,n_state])
  for i in range(n_state):
    for j in range(n_state):
      if j == 0:
        p[i,j] =  np.exp(-(i+1))*np.exp(-(j+1)) + 2*(1-np.exp(-(i+1)))*np.exp(-2*(j+1))
      else:
        p[i,j] = p[i,j-1] + np.exp(-(i+1))*np.exp(-(j+1)) + 2*(1-np.exp(-(i+1)))*np.exp(-2*(j+1)) #
    p[i,:] = p[i,:] / p[i,n_state-1]

  print("p",p)
  traj = np.ndarray([len])
  s = 0
  traj[0] = s
  counter = 0
  for i in range(len):
    s = get_next_state(s,p)
    traj[i] = s + 1
    # if s is not 0:
    #   counter += 1
  # print(counter)
  return traj
    
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
  print(traj)
  return traj

def load_record_csv():
    DATA_DIR = "results/record"
    MAX_FRAME = 25   # maximum frames
    
    filenames = os.listdir(DATA_DIR)[:10000] # only use first 10k episodes
    n = len(filenames)
    print(n, "files loaded!")
    df = pd.DataFrame()
    for j, fname in enumerate(filenames):
        if not fname.endswith('npz'): 
            continue
        file_path = os.path.join(DATA_DIR, fname)

        data = np.load(file_path)
        state = data['obs']
        action = data['action']
        done = data['done']
        
        n_pad = MAX_FRAME - len(state) # pad so they are all a thousand step long episodes
        # print("n_pad: ", n_pad)
        if n_pad > 0:
            state = np.pad(state, (0, n_pad), 'constant')
    return state, action, done

def normalize(rrf):
  norm = np.linalg.norm(rrf)
  norm_rrf = rrf/norm
  return norm_rrf

def Get_p_(p_hat,r,c_12):
  p_ = np.empty_like(p_hat)
  p_hat[0] = np.dot(np.dot(c_12,p_hat[0]),c_12)
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

def Get_P_hat(traj,G, w_orf, w_sorf,step=1):
  avg = np.ndarray((2,n_feature,n_feature))
  i = 0
  counter = 0
  # print(len(traj))
  while i+step < len(traj[0]):
    if i % 1000 == 0:
      print("i",i)
    x = traj[:,i]
  
    y = traj[:,i+step]
    k_orf, k_sorf = Kernel(x,y,G, w_orf, w_sorf)
    
    # avg[0] += k_rff_1
    # avg[0] += k_rff_2
    avg[0] += k_orf
    avg[1] += k_sorf
    i += step
    counter += 1
  avg = avg/counter
  # print("p_hat",avg)
  return avg

def Kernel(x,y,G,w_orf,w_sorf):
  # rff_1_x = RFF_1(x,w,n)
  # rff_1_y = RFF_1(y,w,n)
  # k_rff_1 = np.dot(rff_1_x.transpose(),rff_1_y)
  # print("rff_1_x", rff_1_x)
  # print("rff_1_y.shape", rff_1_y.shape)
  # rff_2_x = RFF_2(x,G)
  # rff_2_y = RFF_2(y,G)
  # k_rff_2 = np.dot(rff_2_x.transpose(),rff_2_y)
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
  return  k_orf, k_sorf  # n * n

# random Fourier feature 
def RFF_1(x,w,n_raw,n_feature):
    rff = np.zeros((n_raw,n_feature))
    rff[0,:]  = np.sqrt(2.0/n_feature)*(np.sin(w*x[0])) 
    # q, r = np.linalg.qr(rff_x)
    # print("q",q)
    # print("r",r)
    rff[1,:] = np.sqrt(2.0/n_feature)*(np.cos(w*x[1]))  # D*1 
    return rff  #D*2

def RFF_2(x,G):
  rff = np.reshape(np.dot(G,x.transpose()),(1,-1))
  rff = normalize(rff)
  return rff # 2*D

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
  # print("norm", norm)
  # orf = np.dot(w_orf,x.transpose())
  return c_diag, w_orf

def ORF(x,w_orf):

  orf = np.reshape(np.dot(w_orf,x.transpose()),(1,-1))
  # print(orf)
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


if __name__ == "__main__":
    # state, action, done = load_record_csv()
    # traj = brownian_motion(10000)
    traj = get_mc_traj(5000)
    print(traj)
    r = 4 # rank
    n_feature_list = [256] # n features
    n_raw = 1 # n-dimensional raw data
    step_list = [1]
   

    # n_list = [16] # n features
    # step_list = [1]
    # x = np.array([[0],[1],[2]])
    # y = np.array([[0],[1],[2]])
    x_list = [np.array([1]),np.array([2]),np.array([3])]
    y_list = [np.array([1]),np.array([2]),np.array([3])]
    for n_feature in n_feature_list:
      for step in step_list:
        print("n:",n_feature)
        print("step:",step)
        # w for rff_1
        w = np.random.normal(loc=0.0, scale=1.0, size=n_feature) # n * 1
        # gaussian for rff_2
        G = np.random.normal(loc=0.0, scale=1.0, size=(n_feature,n_raw))
        c_12 , w_orf = get_w_orf(n_raw,n_feature,G)
        w_sorf = get_w_sorf(n_raw,n_feature)
        p_hat = Get_P_hat(traj, G , w_orf, w_sorf,step)
        # 
        p_ = Get_p_(p_hat,r,c_12)
        
        # # print("p_hat:",p_hat)
        print("p_:",p_)

        for x in x_list:
          for y in y_list:
            orf_x  = ORF(x,w_orf)
            # orf_x = normalize(orf_x)
            orf_y = ORF(y,w_orf)
            # orf_y = normalize(orf_y)
            sorf_x = SORF(x,w_sorf)
            # sorf_x = normalize(sorf_x)
            sorf_y = SORF(y,w_sorf)
            # sorf_y = normalize(sorf_y)

            # p_rff_1 = np.dot(np.dot(rff_1_x, p_[0]),rff_1_y.transpose())
            # p_rff_2 = np.dot(np.dot(rff_2_x, p_[0]),rff_2_y.transpose())
            p_orf = np.dot(np.dot(np.dot(np.dot(orf_x,c_12), p_[0]),c_12),orf_y.transpose())
            p_sorf = np.dot(np.dot(sorf_x, p_[1]),sorf_y.transpose())

            print("x",x)
            print("y",y)
            # print("p_rff_1", p_rff_1)
            # print("p_rff_2 ",p_rff_2)
            print("p_orf" ,p_orf)
            print("p_sorf",p_sorf)
        



  