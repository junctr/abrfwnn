import numpy as np
import time
#import scipy.integrate as sp
from matplotlib import pyplot as plt
from tqdm import tqdm
import csv
from multiprocessing import Pool, cpu_count
from numba import njit

q = np.array([
    [1,2],
    [3,4],
    [5,6]]
)

qd = np.array([
    [10,20],
    [30,40],
    [50,60]]
)

s = np.array([
    [100],
    [300],
    [500]]
)

def x_f0(q, qd, s):
    
    x = np.zeros((15,1), dtype=np.float64)

    # x = np.concatenate([q.T.reshape(-1,1), qd_f(t).T.reshape(-1,1), ddqd_f(t)])
    x = np.concatenate([q.T.reshape(-1,1), qd.T.reshape(-1,1), s])
    
    return x

def x_f1(q, qd, s):

    x = np.zeros((15,1))

    # x = np.concatenate([q.T.copy().reshape(-1,1), qd_f(t).T.copy().reshape(-1,1), ddqd_f(t)])
    # x = np.concatenate([q.T.copy().reshape(-1,1), qd_f(t).T.copy().reshape(-1,1), s])
    
    for i in range(2):
        
        for j in range(3):
            x[i*3+j][0] = q[j][i]
    
    for i in range(2):
        
        for j in range(3):
            x[i*3+j+6][0] = qd[j][i]
    
    for j in range(3):
        
        x[j+12][0] = s[j][0]
        
    return x

# for i in range(2):
    
#     for j in range(3):
#         print(q[j][i])

for i in range(2):
    
    for j in range(3):
        print(j+i*3)
        print(i,j)
        # print(i)
        # print(j)
        print(" ")

print(x_f0(q,qd,s))
print(x_f1(q,qd,s))
print(np.zeros((2,2)))