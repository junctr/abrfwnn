# -*- coding: utf-8 -*-

import numpy as np
import time
#import scipy.integrate as sp
from matplotlib import pyplot as plt
from tqdm import tqdm
import csv


def system(t, q, tau, wn):
    
    dq = np.empty((3, 2), dtype=np.float64)

    #dq = [[q1,q1_dot],[q2,q2_dot],[q3,q3_dot]]
    
    dq[:,[0]] = q[:,[1]]
    dq[:,[1]] = np.linalg.inv(M(q)) @ (tau - tau0_f(t) - wn - np.dot(C(q), q[:,[1]]) - G(q) - F(q))

    return dq

def M(q):

    M = np.zeros((3,3), dtype=np.float64)

    M[0][0] = l[0]**2 * (p[0] + p[1]) + p[1] * (l[1]**2 + 2 * l[0] * l[1] * np.cos(q[1][0]))
    #M[0][0] = l[0]**2 * (p[0] + p[1]) + p[1] * l[1]**2 + 2 * p[0] * p[1] * np.cos(q[1][0])
    M[0][1] = p[1] * l[1]**2 + p[1] * l[0] * l[1] * np.cos(q[1][0])
    M[1][0] = M[0][1]
    M[1][1] = p[1] * l[1]**2
    M[2][2] = p[2]

    return M

def C(q):

    C = np.zeros((3,3), dtype=np.float64)

    C[0][0] = -p[1] * l[0] * l[1] * (2 * q[0][1] * q[1][1] + q[1][1]**2) * np.sin(q[1][0])
    C[1][0] = p[1] * l[0] * l[1] * q[0][1]**2 * np.sin(q[1][0])

    return C

def G(q):
    
    G = np.array([
        [(p[0] + p[1]) * g * l[0] * np.cos(q[0][0]) + p[1] * g * l[1] * np.cos(q[0][0] + q[1][0])],
        [p[1] * g * l[1] * np.cos(q[0][0] + q[1][0])],
        [-p[2] * g]],
        dtype=np.float64
    )

    return G

def F(q):

    F = np.array([
        [5*q[0][1] + 0.2 * np.sign(q[0][1])],
        [5*q[1][1] + 0.2 * np.sign(q[1][1])],
        [5*q[2][1] + 0.2 * np.sign(q[2][1])]],
        dtype=np.float64
    )

    return F
    
def qd_f(t):

    qd = np.array([
        [0.5*np.sin(2*np.pi*t), np.pi*np.cos(2*np.pi*t)], 
        [0.5*np.sin(2*np.pi*t), np.pi*np.cos(2*np.pi*t)], 
        [0.5*np.sin(2*np.pi*t), np.pi*np.cos(2*np.pi*t)]],
        dtype=np.float64
    )

    return qd

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

def dwn_f(wn):
    
    # dwn = np.array([
    #     [0.0],
    #     [0.0],
    #     [0.0]],
    #     dtype=np.float64
    # )
    
    wnv = np.array([
        [np.random.normal()],
        [np.random.normal()],
        [np.random.normal()]],
        dtype=np.float64
    )
    
    dwn = -np.linalg.inv(alpha_wn0 * np.identity(3, dtype=np.float64)) @ wn + alpha_wn1 * np.identity(3, dtype=np.float64) @ wnv
    
    return dwn

def e_f(t, q):

    e = np.empty((3,2), dtype=np.float64)

    e = qd_f(t) - q

    return e

def s_f(e):

    s = np.empty((3,1), dtype=np.float64)

    s = e[:,[1]] + ((5 * np.identity(3, dtype=np.float64)) @ e[:,[0]])

    return s

def x_f(t, q, s):

    x = np.empty((15,1), dtype=np.float64)

    x = np.concatenate([q.T.reshape(-1,1), qd_f(t).T.reshape(-1,1), s])

    return x

def xji_f(x, xold, odot, co, ro):

    xji = np.empty((15,5), dtype=np.float64)

    xji = x + odot * np.exp(A_f(xold,co,ro))

    return xji

def A_f(xji, co, ro):

    A = np.empty((15,5), dtype=np.float64)

    A = -(co**2) * ((xji - ro)**2)

    return A

def mu_f(A):

    mu = np.empty((1,5), dtype=np.float64)

    mu = np.prod((1 + A) * np.exp(A), axis=0)

    #muji = (1 + A) * np.exp(A)

    return mu

def muji_f(A):

    muji = (1 + A) * np.exp(A)

    return muji

def y_f(A, W):

    #y = np.empty((3,1), dtype=np.float64)

    y = W.T @ mu_f(A).reshape(5,1)

    return y

def omega_f(odot, co, ro, W):

    omega = np.array([
        [1],
        [np.linalg.norm(odot)],
        [np.linalg.norm(co)],
        [np.linalg.norm(ro)],
        [np.linalg.norm(W)]],
        dtype=np.float64
    )

    return omega

def taus0_f(s, beta, zeta, omega):

    taus0 = ((beta.T @ omega)**2 / (np.linalg.norm(s) * beta.T @ omega + zeta)) * s

    return taus0

def taus1_f(s):
    
    taus1 = np.array([[alpha_s0,0.0,0.0],[0.0,alpha_s1,0.0],[0.0,0.0,alpha_s2]],dtype=np.float64) @ np.sign(s)
    
    return taus1

def tau_f(s, taus0, taus1, y):
    
    tau = np.empty((3,1), dtype=np.float64)

    K = 100 * np.identity(3, dtype=np.float64)

    tau = taus0 + taus1 + K @ s + y
    # tau = taus0 + K @ s + y
    #tau = y(A,W)
    #tau = taus(s,beta,zeta,omega) + K @ s

    return tau    

def B_f(x, Aold, odot, ro):

    B = x + odot * np.exp(Aold) - ro

    return B    

def bk_f(mu, muji, A, Aold, B, co):

    bk = np.empty((5,75), dtype=np.float64)

    dmuji =(2 + A) * np.exp(A) *(-2 * co**2 * np.exp(Aold) * B)

    #x = mu * dmuji / muji
    x = mu * np.divide(dmuji, muji, out=np.zeros_like(dmuji), where=muji!=0)

    zeros0 = np.zeros((15,5), dtype=np.float64) 
    zeros1 = np.zeros((15,5), dtype=np.float64) 
    zeros2 = np.zeros((15,5), dtype=np.float64) 
    zeros3 = np.zeros((15,5), dtype=np.float64) 
    zeros4 = np.zeros((15,5), dtype=np.float64) 

    zeros0[:,[0]] = x[:,[0]]
    zeros1[:,[1]] = x[:,[1]]
    zeros2[:,[2]] = x[:,[2]]
    zeros3[:,[3]] = x[:,[3]]
    zeros4[:,[4]] = x[:,[4]]

    bk[0] = zeros0.T.reshape(1,-1)
    bk[1] = zeros1.T.reshape(1,-1)
    bk[2] = zeros2.T.reshape(1,-1)
    bk[3] = zeros3.T.reshape(1,-1)
    bk[4] = zeros4.T.reshape(1,-1)

    return bk.T

def ek_f(mu, muji, A, Aold, B ,odot, co, ro, xold):

    ek = np.empty((5,75), dtype=np.float64)

    dmuji =(2 + A) * np.exp(A) *(-2 * co * B**2 -2 * co**2 * B *(-2 * odot * co * (xold - ro)**2 ) * np.exp(Aold))

    #x = mu * dmuji / muji
    x = mu * np.divide(dmuji, muji, out=np.zeros_like(dmuji), where=muji!=0)

    zeros0 = np.zeros((15,5), dtype=np.float64) 
    zeros1 = np.zeros((15,5), dtype=np.float64) 
    zeros2 = np.zeros((15,5), dtype=np.float64) 
    zeros3 = np.zeros((15,5), dtype=np.float64) 
    zeros4 = np.zeros((15,5), dtype=np.float64) 

    zeros0[:,[0]] = x[:,[0]]
    zeros1[:,[1]] = x[:,[1]]
    zeros2[:,[2]] = x[:,[2]]
    zeros3[:,[3]] = x[:,[3]]
    zeros4[:,[4]] = x[:,[4]]

    ek[0] = zeros0.T.reshape(1,-1)
    ek[1] = zeros1.T.reshape(1,-1)
    ek[2] = zeros2.T.reshape(1,-1)
    ek[3] = zeros3.T.reshape(1,-1)
    ek[4] = zeros4.T.reshape(1,-1)

    return ek.T

def gk_f(mu, muji, A, Aold, B ,odot, co, ro):

    gk = np.empty((5,75), dtype=np.float64)

    dmuji =(2 + A) * np.exp(A) *(-2 * co**2 * B * (-1 -2 * odot * co**2 * ro * np.exp(Aold)))

    #x = mu * dmuji / muji
    x = mu * np.divide(dmuji, muji, out=np.zeros_like(dmuji), where=muji!=0)

    zeros0 = np.zeros((15,5), dtype=np.float64) 
    zeros1 = np.zeros((15,5), dtype=np.float64) 
    zeros2 = np.zeros((15,5), dtype=np.float64) 
    zeros3 = np.zeros((15,5), dtype=np.float64) 
    zeros4 = np.zeros((15,5), dtype=np.float64) 

    zeros0[:,[0]] = x[:,[0]]
    zeros1[:,[1]] = x[:,[1]]
    zeros2[:,[2]] = x[:,[2]]
    zeros3[:,[3]] = x[:,[3]]
    zeros4[:,[4]] = x[:,[4]]

    gk[0] = zeros0.T.reshape(1,-1)
    gk[1] = zeros1.T.reshape(1,-1)
    gk[2] = zeros2.T.reshape(1,-1)
    gk[3] = zeros3.T.reshape(1,-1)
    gk[4] = zeros4.T.reshape(1,-1)

    return gk.T


n_seed = 4
np.random.seed(n_seed)
print(n_seed)

alpha_w = 50 * np.identity(5, dtype=np.float64)
alpha_odot = 20 * np.identity(75, dtype=np.float64)
alpha_co = 20 * np.identity(75, dtype=np.float64)
alpha_ro = 20 * np.identity(75, dtype=np.float64)
alpha_beta = 0.001 * np.identity(5, dtype=np.float64)
alpha_zeta = 0.1
alpha_lambda = 0.3
alpha_wn0 = 100
alpha_wn1 = 10
alpha_s0 = 5.0
alpha_s1 = 5.0
alpha_s2 = 5.0

zeta = 1
omega = np.ones((5,1), dtype=np.float64)
beta = 0.1 * np.array([
    [1],
    [1],
    [1],
    [1],
    [1]],
    dtype=np.float64
)
wn = np.array([
    [0.0],
    [0.0],
    [0.0]],
    dtype=np.float64
)

p = np.array([4, 3, 1.5])
l = np.array([0.4, 0.3, 0.2])
g = 10

t = 0.0
end = 100
step = 0.0001
i = 0

m = -0.01
n = 1.01
q = np.array([
    [m * 0.5, n * np.pi],
    [m * 0.5, n * np.pi],
    [m * 0.5, n * np.pi]],
    dtype=np.float64
)
T = 1000
xold0 = np.array([[m*0.5,m*0.5,m*0.5,n*np.pi,n*np.pi,n*np.pi,0,0,0,np.pi,np.pi,np.pi,(1-n)*np.pi-m*2.5,(1-n)*np.pi-m*2.5,(1-n)*np.pi-m*2.5]], dtype=np.float64).reshape(-1,1)
xold = [xold0 for i_xold in range(T)]

W = 50 * 2 * (np.random.rand(5,3) - 0.5)
j_q = 1.0 * 0.5
j_dq = 1.0 * np.pi
j_s = 0.1 * 1.0 * np.pi * np.sqrt(2)
j = np.array([[j_q,j_q,j_q,j_dq,j_dq,j_dq,j_q,j_q,j_q,j_dq,j_dq,j_dq,j_s,j_s,j_s]]).T
odot = j * 0.1 * 2 * (np.random.rand(15,5) - 0.5)
co = (1/j) * 0.5 * 2 * (np.random.rand(15,5) - 0.5)
ro = j * 1 * 2 * (np.random.rand(15,5) - 0.5)

Wold = []
odotold = []
coold = []
roold = []
Wold.append(W.copy())
odotold.append(odot.copy())
coold.append(co.copy())
roold.append(ro.copy())

print("W")
print(np.round(W,4))
print("odot")
print(np.round(odot/j,4))
print("co")
print(np.round(co*j,4))
print("ro")
print(np.round(ro/j,4))
print("beta")
print(beta)
print("zeta")
print(zeta)

e_0 = []
e_1 = []
e_2 = []
e_3 = []
e_4 = []
e_5 = []
e_6 = []
e_7 = []
e_8 = []
e_9 = []
e_10 = []
e_11 = []
e_12 = []
e_13 = []
e_14 = []
e_15 = []
e_16 = []
e_17 = []
e_18 = []
e_19 = []
e_20 = []
e_21 = []
e_22 = []
e_23 = []
e_24 = []
e_25 = []
e_26 = []
e_27 = []

t_data = []

start = time.time()

for i in tqdm(range(int(end/step))):

    qd = qd_f(t)
    e = e_f(t,q)
    s = s_f(e)
    x = x_f(t,q,s)
    xji = xji_f(x,xold[-T],odot,co,ro)
    A = A_f(xji,co,ro)
    Aold = A_f(xold[-T], co,ro)
    B = B_f(x,Aold,odot,ro)
    mu = mu_f(A)
    muji = muji_f(A)
    omega = omega_f(odot,co,ro,W)
    y = y_f(A,W)
    taus0 = taus0_f(s,beta,zeta,omega)
    taus1 = taus1_f(s)
    tau = tau_f(s,taus0,taus1,y)
    bk = bk_f(mu,muji,A,Aold,B,co)
    ek = ek_f(mu,muji,A,Aold,B,odot,co,ro,xold[-1])
    gk = gk_f(mu,muji,A,Aold,B,odot,co,ro)
    
    if zeta > 0.1:
    
        k_beta = np.linalg.norm(s) * alpha_beta @ omega
        k_zeta = -alpha_zeta * zeta
    
    else :
        
        k_beta = 0.0
        k_zeta = 0.0
    
    dwn = dwn_f(wn)
    
    k1_q = system(t,q,tau,wn)
    k2_q = system(t+step/2,q+(step/2)*k1_q,tau,wn)
    k3_q = system(t+step/2,q+(step/2)*k2_q,tau,wn)
    k4_q = system(t+step,q+step*k3_q,tau,wn)

    xold.append(x.copy())

    Wold.append(W.copy())
    odotold.append(odot.copy())
    coold.append(co.copy())
    roold.append(ro.copy())
    #betaold.append(beta.copy())
    #zetaold.append(zeta)

    if i%10 == 0:
            
            e_0.append(e[0][0])
            e_1.append(e[1][0])
            e_2.append(e[2][0])
            e_3.append(tau[0][0])
            e_4.append(tau[1][0])
            e_5.append(tau[2][0])
            e_6.append(taus0[0][0])
            e_7.append(taus0[1][0])
            e_8.append(taus0[2][0])
            e_9.append((100 * np.identity(3, dtype=np.float64)@s)[0][0])
            e_10.append((100 * np.identity(3, dtype=np.float64)@s)[1][0])
            e_11.append((100 * np.identity(3, dtype=np.float64)@s)[2][0])
            e_12.append(y[0][0])
            e_13.append(y[1][0])
            e_14.append(y[2][0])
            e_15.append(e[0][1])
            e_16.append(e[1][1])
            e_17.append(e[2][1])
            e_18.append(s[0][0])
            e_19.append(s[1][0])
            e_20.append(s[2][0])
            e_21.append(wn[0][0])
            e_22.append(wn[1][0])
            e_23.append(wn[2][0])
            e_24.append(taus1[0][0])
            e_25.append(taus1[1][0])
            e_26.append(taus1[2][0])
            e_27.append(beta.T @ omega)

            t_data.append(t)

    # if i%1000 == 0:
    #     print(round(100*t/end, 1),"%",i,round(time.time()-start, 1),"s")

    q += (step / 6) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
    W += step * (alpha_w @ (mu.reshape(5,1) - bk.T @ odot.T.reshape(-1,1) - ek.T @ co.T.reshape(-1,1) - gk.T @ ro.T.reshape(-1,1)) @ s.T) + alpha_lambda * (Wold[-1] - Wold[-2])
    odot += step * ((alpha_odot @ bk @ W @ s).reshape(5,15).T) + alpha_lambda * (odotold[-1] - odotold[-2])
    co += step * ((alpha_co @ ek @ W @ s).reshape(5,15).T) + alpha_lambda * (coold[-1] - coold[-2])
    ro += step * ((alpha_ro @ gk @ W @ s).reshape(5,15).T) + alpha_lambda * (roold[-1] - roold[-2])
    beta += step * k_beta
    zeta += step * k_zeta
    wn += step * dwn
    
    t += step
    i += 1
    

print("W")
print(np.round(W,4))
print("odot")
print(np.round(odot/j,4))
print("co")
print(np.round(co*j,4))
print("ro")
print(np.round(ro/j,4))
print("beta")
print(beta)
print("zeta")
print(zeta)

e_all = [
    e_0,
    e_1,
    e_2,
    e_3,
    e_4,
    e_5,
    e_6,
    e_7,
    e_8,
    e_9,
    e_10,
    e_11,
    e_12,
    e_13,
    e_14,
    e_15,
    e_16,
    e_17,
    e_18,
    e_19,
    e_20,
    e_21,
    e_22,
    e_23,
    e_24,
    e_25,
    e_26,
    e_27,
]

param_all = [odot,co,ro,W,beta,zeta]
#param_all_old = [odotold,coold,roold,Wold,betaold,zetaold]

print("odot_j")
print(np.round(odot/j,4))
print("co_j")
print(np.round(co*j,4))
print("ro_j")
print(np.round(ro/j,4))

print("W")
print(np.round(W,4))
print("odot")
print(np.round(odot,4))
print("co")
print(np.round(co,4))
print("ro")
print(np.round(ro,4))
print("beta")
print(beta)
print("zeta")
print(zeta)

#print(e_all.shape)
#print(param_all[0].shape)
print("n_data")
print(len(t_data))

np.save(f"data/p_s{n_seed}_m{alpha_lambda}_wn{alpha_wn0}_{alpha_wn1}_s{alpha_s0}_{alpha_s1}_{alpha_s2}_T{T}_step{step}_t{end}_param_all.npy",param_all)
#np.save(f"k_s{n_seed}_m{alpha_lambda}_T{T}_t{end}_param_all_old.npy",param_all_old)
np.savetxt(f"data/p_s{n_seed}_m{alpha_lambda}_wn{alpha_wn0}_{alpha_wn1}_s{alpha_s0}_{alpha_s1}_{alpha_s2}_T{T}_step{step}_t{end}_e_all.csv",e_all)