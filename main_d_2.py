# -*- coding: utf-8 -*-
# rfwnn
# 0 system
# D
# no wn
# no sign
# beta zeta D stop

from func_2 import *
import numpy as np
import time
#import scipy.integrate as sp
from matplotlib import pyplot as plt
from tqdm import tqdm
import csv
from multiprocessing import Pool, cpu_count
import os
from numba import njit

def main(n_seed: int):
    
    np.random.seed(n_seed)
    # # print(f"START n_seed={n_seed}")
    
    t = 0.0
    i = 0
    
    end = 100
    step = 0.0001

    alpha_w = 50 * np.identity(5)
    alpha_v = 20 * np.identity(75)
    alpha_a = 20 * np.identity(75)
    alpha_b = 20 * np.identity(75)
    alpha_beta = 0.001 * np.identity(5)
    alpha_zeta = 0.1
    alpha_lambda = 0.3
    alpha_d = 10 * np.identity(3)
    alpha_dk = 0.2 * np.identity(3)
    alpha_wn0 = 100
    alpha_wn1 = 10
    alpha_s0 = 5.0
    alpha_s1 = 5.0
    alpha_s2 = 5.0

    T = 1000

    p = np.array([4, 3, 1.5])
    l = np.array([0.4, 0.3, 0.2])
    g = 10
    
    zeta = 1
    omega = np.ones((5,1))
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

    m = -0.01
    n = 1.01
    q = np.array([
        [m * 0.5, n * np.pi],
        [m * 0.5, n * np.pi],
        [m * 0.5, n * np.pi]]
    )
    # xold0 = np.array([[m*0.5,m*0.5,m*0.5,n*np.pi,n*np.pi,n*np.pi,0,0,0,np.pi,np.pi,np.pi,0,0,0]]).reshape(-1,1)
    xold0 = np.array([[m*0.5,m*0.5,m*0.5,n*np.pi,n*np.pi,n*np.pi,0,0,0,np.pi,np.pi,np.pi,(1-n)*np.pi-m*2.5,(1-n)*np.pi-m*2.5,(1-n)*np.pi-m*2.5]]).reshape(-1,1)
    xold = [xold0 for i_xold in range(T)]
    
    # xold = []
    # for i_xold in range(T):
    #     xold.append(xold0)

    W = 50 * 2 * (np.random.rand(5,3) - 0.5)
    j_q = 1.0 * 0.5
    j_dq = 1.0 * np.pi
    # j_ddq = 2.0 * np.pi**2
    j_s = 0.1 * 1.0 * np.pi * np.sqrt(2)
    # j = np.array([[j_q,j_q,j_q,j_dq,j_dq,j_dq,j_q,j_q,j_q,j_dq,j_dq,j_dq,j_ddq,j_ddq,j_ddq]]).T
    j = np.array([[j_q,j_q,j_q,j_dq,j_dq,j_dq,j_q,j_q,j_q,j_dq,j_dq,j_dq,j_s,j_s,j_s]]).T
    v = j * 0.1 * 2 * (np.random.rand(15,5) - 0.5)
    a = (1/j) * 0.5 * 2 * (np.random.rand(15,5) - 0.5)
    b = j * 1 * 2 * (np.random.rand(15,5) - 0.5)
    D = np.array([
        [1.0,0.0,0.0],
        [0.0,1.0,0.0],
        [0.0,0.0,1.0]],
        dtype=np.float64
    )
    
    Wold = []
    vold = []
    aold = []
    bold = []
    Dold = []
    Wold.append(W.copy())
    vold.append(v.copy())
    aold.append(a.copy())
    bold.append(b.copy())
    Dold.append(D.copy())

    # print("W")
    # print(np.round(W,4))
    # print("v_j")
    # print(np.round(v/j,4))
    # print("a_j")
    # print(np.round(a*j,4))
    # print("b_j")
    # print(np.round(b/j,4))
    # print("D")
    # print(np.round(D,4))
    # print("beta")
    # print(beta)
    # print("zeta")
    # print(zeta)

    # print(f"NOW n_seed={n_seed}")
    # print(f"abrfwnn/data_test_old_d/p_bzd_s{n_seed}_m{alpha_lambda}_wn{alpha_wn0}_{alpha_wn1}_s{alpha_s0}_{alpha_s1}_{alpha_s2}_T{T}_step{step}_t{end}_param_all.npy")

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
    e_28 = []
    e_29 = []
    e_30 = []
    e_31 = []
    e_32 = []
    e_33 = []

    t_data = []

    # start = time.time()

    for i in tqdm(range(int(end/step))):

        qd = qd_f(t)
        e = e_f(t,q)
        s = s_f(e)
        x = x_f(q,qd,s)
        xji = xji_f(x,xold[-T],v,a,b)
        A = A_f(xji,a,b)
        Aold = A_f(xold[-T], a,b)
        B = B_f(x,Aold,v,b)
        muji = muji_f(A)
        mu = mu_f(muji)
        omega = omega_f(v,a,b,W)
        y = y_f(muji,W)
        taus0 = taus0_f(s,beta,zeta,omega)
        taus1 = taus1_f(s,alpha_s0,alpha_s1,alpha_s2)
        taud = taud_f(e, s,taus0,taus1,y)
        tau = tau_f(taud, D)
        vk = vk_f(mu,muji,A,Aold,B,a)
        ak = ak_f(mu,muji,A,Aold,B,v,a,b,xold[-1])
        bk = bk_f(mu,muji,A,Aold,B,v,a,b)

        if zeta > 0.1:

            k_beta = np.linalg.norm(s) * alpha_beta @ omega
            k_zeta = -alpha_zeta * zeta
            if taud[0] > 0:
                
                k_D = alpha_d @ np.ones((3,1)) @ s.T - alpha_d @ alpha_dk @ D * np.linalg.norm(s)

            elif taud[0] < 0:

                k_D = alpha_d @ -np.ones((3,1)) @ s.T - alpha_d @ alpha_dk @ D * np.linalg.norm(s)
        
            else :

                k_D = - alpha_d @ alpha_dk @ D * np.linalg.norm(s)

        else :

            k_beta = 0.0
            k_zeta = 0.0
            k_D = np.zeros((3,3))

        # k_beta = np.linalg.norm(s) * alpha_beta @ omega
        # k_zeta = -alpha_zeta * zeta
        
        # if taud[0] > 0:
            
        #     k_D = alpha_d @ np.ones((3,1)) @ s.T - alpha_d @ alpha_dk @ D * np.linalg.norm(s)

        # elif taud[0] < 0:
            
        #     k_D = alpha_d @ -np.ones((3,1)) @ s.T - alpha_d @ alpha_dk @ D * np.linalg.norm(s)
        
        # else :
            
        #     k_D = - alpha_d @ alpha_dk @ D * np.linalg.norm(s)

        dwn = dwn_f(wn)

        k1_q = system(t,q,tau,wn,p,l,g)
        k2_q = system(t+step/2,q+(step/2)*k1_q,tau,wn,p,l,g)
        k3_q = system(t+step/2,q+(step/2)*k2_q,tau,wn,p,l,g)
        k4_q = system(t+step,q+step*k3_q,tau,wn,p,l,g)

        xold.append(x.copy())

        Wold.append(W.copy())
        vold.append(v.copy())
        aold.append(a.copy())
        bold.append(b.copy())
        Dold.append(D.copy())
        #betaold.append(beta.copy())
        #zetaold.append(zeta)

        if i%10 == 0:

                e_0.append(e[0][0])
                e_1.append(e[1][0])
                e_2.append(e[2][0])
                e_3.append(tau[0][0])
                e_4.append(tau[1][0])
                e_5.append(tau[2][0])
                e_6.append(taud[0][0])
                e_7.append(taud[1][0])
                e_8.append(taud[2][0])
                e_9.append(y[0][0])
                e_10.append(y[1][0])
                e_11.append(y[2][0])
                e_12.append(taus0[0][0])
                e_13.append(taus0[1][0])
                e_14.append(taus0[2][0])
                e_15.append((100 * np.identity(3)@s)[0][0])
                e_16.append((100 * np.identity(3)@s)[1][0])
                e_17.append((100 * np.identity(3)@s)[2][0])
                e_18.append(taus1[0][0])
                e_19.append(taus1[1][0])
                e_20.append(taus1[2][0])
                e_21.append(wn[0][0])
                e_22.append(wn[1][0])
                e_23.append(wn[2][0])
                e_24.append(e[0][1])
                e_25.append(e[1][1])
                e_26.append(e[2][1])
                e_27.append(s[0][0])
                e_28.append(s[1][0])
                e_29.append(s[2][0])
                e_30.append(D[0][0])
                e_31.append(D[1][1])
                e_32.append(D[2][2])
                e_33.append(beta.T @ omega)

                t_data.append(t)

        # if i%1000 == 0:
        #     # print(round(100*t/end, 1),"%",i,round(time.time()-start, 1),"s")

        q += (step / 6) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
        W += step * (alpha_w @ (mu.reshape(5,1) - vk.T @ v.T.reshape(-1,1) - ak.T @ a.T.reshape(-1,1) - bk.T @ b.T.reshape(-1,1)) @ s.T) + alpha_lambda * (Wold[-1] - Wold[-2])
        v += step * ((alpha_v @ vk @ W @ s).reshape(5,15).T) + alpha_lambda * (vold[-1] - vold[-2])
        a += step * ((alpha_a @ ak @ W @ s).reshape(5,15).T) + alpha_lambda * (aold[-1] - aold[-2])
        b += step * ((alpha_b @ bk @ W @ s).reshape(5,15).T) + alpha_lambda * (bold[-1] - bold[-2])
        beta += step * k_beta
        zeta += step * k_zeta

        D += step * k_D + alpha_lambda * (Dold[-1] - Dold[-2])

        wn += step * dwn

        t += step
        i += 1

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
        e_28,
        e_29,
        e_30,
        e_31,
        e_32,
        e_33,
    ]

    param_all = [v,a,b,W,D,beta,zeta]

    # # print("W")
    # # print(np.round(W,4))
    # # print("v_j")
    # # print(np.round(v/j,4))
    # # print("a_j")
    # # print(np.round(a*j,4))
    # # print("b_j")
    # # print(np.round(b/j,4))
    # # print("D")
    # # print(np.round(D,4))
    # # print("beta")
    # # print(beta)
    # # print("zeta")
    # # print(zeta)

    # ## print(e_all.shape)
    # ## print(param_all[0].shape)
    # # print("n_data")
    # # print(len(t_data))

    dir_base = "data_test_old_d/"
    os.makedirs(dir_base, exist_ok=True)
    np.save(dir_base + f"p_bzd_s{n_seed}_m{alpha_lambda}_wn{alpha_wn0}_{alpha_wn1}_s{alpha_s0}_{alpha_s1}_{alpha_s2}_T{T}_step{step}_t{end}_param_all.npy",param_all)
    # #np.save(f"k_s{n_seed}_m{alpha_lambda}_T{T}_t{end}_param_all_old.npy",param_all_old)
    np.savetxt(dir_base + f"p_bzd_s{n_seed}_m{alpha_lambda}_wn{alpha_wn0}_{alpha_wn1}_s{alpha_s0}_{alpha_s1}_{alpha_s2}_T{T}_step{step}_t{end}_e_all.csv",e_all,delimiter = ",")

    # print(f"END n_seed={n_seed}")

# if __name__ == '__main__':
    
#     start = time.perf_counter()
    
#     print(cpu_count())

#     init = []

#     for i in range(1):
        
#         init.append((i,))

#     with Pool(cpu_count()) as p:
        
#         r = p.starmap(func=main,iterable=init)
        
#     print(time.perf_counter() - start)

if __name__ == '__main__':
    
    main(0)