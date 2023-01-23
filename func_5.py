# -*- coding: utf-8 -*-
# rfwnn
# 0 system
# D
# no wn
# no sign
# beta zeta D stop


import numpy as np
import time
#import scipy.integrate as sp
from matplotlib import pyplot as plt
from tqdm import tqdm
import csv
from multiprocessing import Pool, cpu_count
from numba import njit

@njit(cache=True)
def M(q, p, l):

    M = np.zeros((3,3))

    M[0][0] = l[0]**2 * (p[0] + p[1]) + p[1] * (l[1]**2 + 2 * l[0] * l[1] * np.cos(q[1][0]))
    M[0][1] = p[1] * l[1]**2 + p[1] * l[0] * l[1] * np.cos(q[1][0])
    M[1][0] = M[0][1]
    M[1][1] = p[1] * l[1]**2
    M[2][2] = p[2]
    
    # M[0][0] = l[0]**2 * (p[0]/3 + p[1] + p[2]) + l[0] * l[1] * (p[1] + 2 * p[2]) * np.cos(q[1][0]) + l[1]**2 * (p[1]/3 + p[2])
    # M[0][1] = -l[0] * l[1] * (p[1]/3 + p[2]) * np.cos(q[1][0]) - l[1]**2 * (p[1]/3 + p[2])
    # M[1][0] = M[0][1]
    # M[1][1] = l[1]**2 * (p[1]/3 + p[2])
    # M[2][2] = p[2]

    return M

@njit(cache=True)
def C(q, p, l):

    C = np.zeros((3,3))

    C[0][0] = -p[1] * l[0] * l[1] * (2 * q[0][1] * q[1][1] + q[1][1]**2) * np.sin(q[1][0])
    C[1][0] = p[1] * l[0] * l[1] * q[0][1]**2 * np.sin(q[1][0])
    
    # C[0][0] = -q[1][1] * (p[1] + 2 * p[2]) 
    # C[1][0] = p[1] * l[0] * l[1] * q[0][1]**2 * np.sin(q[1][0])
    # C[0][1] = C[1][0]

    return C

@njit(cache=True)
def G(q, p, l, g):
    
    G = np.array([
        [(p[0] + p[1]) * g * l[0] * np.cos(q[0][0]) + p[1] * g * l[1] * np.cos(q[0][0] + q[1][0])],
        [p[1] * g * l[1] * np.cos(q[0][0] + q[1][0])],
        [-p[2] * g]]
    )
    
    # G = np.array([
    #     [-p[2] * g],
    #     [-p[2] * g],
    #     [-p[2] * g]],
    #     dtype=np.float64
    # )

    return G

@njit(cache=True)
def F(q):

    F = np.array([
        [5*q[0][1] + 0.2 * np.sign(q[0][1])],
        [5*q[1][1] + 0.2 * np.sign(q[1][1])],
        [5*q[2][1] + 0.2 * np.sign(q[2][1])]]
    )

    return F

@njit(cache=True)
def system(t, q, tau, wn, p, l, g):
    
    dq = np.zeros((3, 2))

    #dq = [[q1,q1_dot],[q2,q2_dot],[q3,q3_dot]]
    
    dq[:,0:1] = q[:,1:2]
    # dq[:,1:2] = np.linalg.inv(M(q)) @ (tau - tau0_f(t) - wn - np.dot(C(q), q[:,1:2]) - G(q) - F(q))
    dq[:,1:2] = np.linalg.inv(M(q,p,l)) @ (tau - tau0_f(t) - np.dot(C(q,p,l), q[:,1:2]) - G(q,p,l,g) - F(q))

    return dq

@njit(cache=True)
def qd_f(t):

    qd = np.array([
        [0.5*np.sin(2*np.pi*t), np.pi*np.cos(2*np.pi*t)], 
        [0.5*np.sin(2*np.pi*t), np.pi*np.cos(2*np.pi*t)], 
        [0.5*np.sin(2*np.pi*t), np.pi*np.cos(2*np.pi*t)]]
    )

    return qd

@njit(cache=True)
def ddqd_f(t):
    
    ddqd = np.array([
        [-2 * np.pi**2 * np.sin(2*np.pi*t)], 
        [-2 * np.pi**2 * np.sin(2*np.pi*t)], 
        [-2 * np.pi**2 * np.sin(2*np.pi*t)]]
    )

    return ddqd

@njit(cache=True)
def tau0_f(t):
    
    tau0 = np.array([
        [2*np.sin(2*np.pi*t)],
        [2*np.sin(2*np.pi*t)],
        [2*np.sin(2*np.pi*t)]],
        dtype=np.float64
    )
    
    # tau0 = np.array([
    #     [2*np.sin(10 *2*np.pi*t)],
    #     [2*np.sin(10 *2*np.pi*t)],
    #     [2*np.sin(10 *2*np.pi*t)]],
    #     dtype=np.float64
    # )
    
    return tau0

@njit(cache=True)
def dwn_f(wn):
    
    dwn = np.zeros((3,1))
    
    # wnv = np.array([
    #     [np.random.normal()],
    #     [np.random.normal()],
    #     [np.random.normal()]],
    #     dtype=np.float64
    # )
    
    # dwn = -np.linalg.inv(alpha_wn0 * np.identity(3)) @ wn + alpha_wn1 * np.identity(3) @ wnv
    
    return dwn

@njit(cache=True)
def e_f(t, q):

    e = np.zeros((3,2))

    e = qd_f(t) - q

    return e

@njit(cache=True)
def s_f(e):

    s = np.zeros((3,1))

    s = e[:,1:2] + ((5 * np.identity(3)) @ e[:,0:1])
    
    # s = e[:,1:2] + ((6 * np.identity(3)) @ e[:,0:1])

    return s

@njit(cache=True)
def x_f(q, qd, s):

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

@njit(cache=True)
def xji_f(x, xold, v, a, b):

    xji = np.zeros((15,5))

    xji = x + v * np.exp(A_f(xold,a,b))

    return xji

@njit(cache=True)
def A_f(xji, a, b):

    A = np.zeros((15,5))

    A = -(a**2) * ((xji - b)**2)

    return A

@njit(cache=True)
def mu_f(muji):

    mu = np.ones((1,5))

    # mu = np.prod((1 + A) * np.exp(A), axis=0)

    # muji = (1 + A) * np.exp(A)
    
    for i in range(5):
        
        for j in range(15):
        
            mu[0][i] *= muji[j][i]

    return mu

@njit(cache=True)
def muji_f(A):

    muji = (1 + A) * np.exp(A)

    return muji

@njit(cache=True)
def y_f(muji, W):

    #y = np.zeros((3,1))

    y = W.T @ mu_f(muji).copy().reshape(5,1)

    return y

@njit(cache=True)
def omega_f(v, a, b, W):

    omega = np.array([
        [1.0],
        [np.linalg.norm(v)],
        [np.linalg.norm(a)],
        [np.linalg.norm(b)],
        [np.linalg.norm(W)]]
    )

    return omega

@njit(cache=True)
def taus0_f(s, beta, zeta, omega):

    taus0 = ((beta.T @ omega)**2 / (np.linalg.norm(s) * beta.T @ omega + zeta)) * s

    return taus0

@njit(cache=True)
def taus1_f(s, alpha_s0, alpha_s1, alpha_s2):
    
    taus1 = np.array([[alpha_s0,0.0,0.0],[0.0,alpha_s1,0.0],[0.0,0.0,alpha_s2]],dtype=np.float64) @ np.sign(s)
    
    return taus1

@njit(cache=True)
def taud_f(e, s, taus0, taus1, y):
    
    taud = np.zeros((3,1))

    K = 100 * np.identity(3)

    # taud = taus0 + taus1 + K @ s + y
    taud = taus0 + K @ s + y
    # taud = y(A,W)
    # taud = taus(s,beta,zeta,omega) + K @ s
    # taud = taus0 + K @ s + e[:,0:1] + y

    return taud

@njit(cache=True)
def tau_f(taud, D):
    
    # Dtrue = np.array([
    #     [2.0,0.0,0.0],
    #     [0.0,2.0,0.0],
    #     [0.0,0.0,2.0]],
    #     dtype=np.float64
    # )
    
    # Dtrue = np.array([
    #     [2.0],
    #     [2.0],
    #     [2.0]],
    #     dtype=np.float64
    # )
    
    Dtilde = np.array([
        [2.0 - D[0][0]],
        [2.0 - D[1][1]],
        [2.0 - D[2][2]]],
        dtype=np.float64
    )
    
    tau = np.zeros((3,1))
    
    if taud[0] > 0:
        
        tau = taud - Dtilde
    
    elif taud[0] < 0:
        
        tau = taud + Dtilde
    
    tau = taud
    
    return tau

@njit(cache=True)
def B_f(x, Aold, v, b):

    B = x + v * np.exp(Aold) - b

    return B

@njit(cache=True)
def vk_f(mu, muji, A, Aold, B, a):

    vk = np.zeros((5,75))

    dmuji =(2 + A) * np.exp(A) *(-2 * a**2 * np.exp(Aold) * B)
    
    x = np.zeros((15,5))
    
    for i in range(15):
        
        for j in range(5):
            
            if muji[i][j] == 0.0:
                
                pass
                
            else :
                
                x[i][j] = mu[0][j] * dmuji[i][j] / muji[i][j]

    zeros0 = np.zeros((15,5)) 
    zeros1 = np.zeros((15,5)) 
    zeros2 = np.zeros((15,5)) 
    zeros3 = np.zeros((15,5)) 
    zeros4 = np.zeros((15,5)) 

    zeros0[:,0:1] = x[:,0:1]
    zeros1[:,1:2] = x[:,1:2]
    zeros2[:,2:3] = x[:,2:3]
    zeros3[:,3:4] = x[:,3:4]
    zeros4[:,4:5] = x[:,4:5]

    vk[0:1,:] = zeros0.T.copy().copy().reshape(1,75)
    vk[1:2,:] = zeros1.T.copy().copy().reshape(1,75)
    vk[2:3,:] = zeros2.T.copy().copy().reshape(1,75)
    vk[3:4,:] = zeros3.T.copy().copy().reshape(1,75)
    vk[4:5,:] = zeros4.T.copy().copy().reshape(1,75)

    return vk.T

@njit(cache=True)
def ak_f(mu, muji, A, Aold, B ,v, a, b, xold):

    ak = np.zeros((5,75))

    dmuji =(2 + A) * np.exp(A) *(-2 * a * B**2 -2 * a**2 * B *(-2 * v * a * (xold - b)**2 ) * np.exp(Aold))

    x = np.zeros((15,5))
    
    for i in range(15):
        
        for j in range(5):
            
            if muji[i][j] == 0.0:
                
                pass
                
            else :
                
                x[i][j] = mu[0][j] * dmuji[i][j] / muji[i][j]

    zeros0 = np.zeros((15,5)) 
    zeros1 = np.zeros((15,5)) 
    zeros2 = np.zeros((15,5)) 
    zeros3 = np.zeros((15,5)) 
    zeros4 = np.zeros((15,5)) 

    zeros0[:,0:1] = x[:,0:1]
    zeros1[:,1:2] = x[:,1:2]
    zeros2[:,2:3] = x[:,2:3]
    zeros3[:,3:4] = x[:,3:4]
    zeros4[:,4:5] = x[:,4:5]

    ak[0] = zeros0.T.copy().copy().reshape(1,-1)
    ak[1] = zeros1.T.copy().copy().reshape(1,-1)
    ak[2] = zeros2.T.copy().copy().reshape(1,-1)
    ak[3] = zeros3.T.copy().copy().reshape(1,-1)
    ak[4] = zeros4.T.copy().copy().reshape(1,-1)

    return ak.T

@njit(cache=True)
def bk_f(mu, muji, A, Aold, B ,v, a, b):

    bk = np.zeros((5,75))

    dmuji =(2 + A) * np.exp(A) *(-2 * a**2 * B * (-1 -2 * v * a**2 * b * np.exp(Aold)))

    x = np.zeros((15,5))
    
    for i in range(15):
        
        for j in range(5):
            
            if muji[i][j] == 0.0:
                
                pass
                
            else :
                
                x[i][j] = mu[0][j] * dmuji[i][j] / muji[i][j]

    zeros0 = np.zeros((15,5)) 
    zeros1 = np.zeros((15,5)) 
    zeros2 = np.zeros((15,5)) 
    zeros3 = np.zeros((15,5)) 
    zeros4 = np.zeros((15,5)) 

    zeros0[:,0:1] = x[:,0:1]
    zeros1[:,1:2] = x[:,1:2]
    zeros2[:,2:3] = x[:,2:3]
    zeros3[:,3:4] = x[:,3:4]
    zeros4[:,4:5] = x[:,4:5]

    bk[0] = zeros0.T.copy().copy().reshape(1,-1)
    bk[1] = zeros1.T.copy().copy().reshape(1,-1)
    bk[2] = zeros2.T.copy().copy().reshape(1,-1)
    bk[3] = zeros3.T.copy().copy().reshape(1,-1)
    bk[4] = zeros4.T.copy().copy().reshape(1,-1)

    return bk.T