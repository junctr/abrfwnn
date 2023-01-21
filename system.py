import fazi_decision_center_Kp as cenKp
import fazi_decision_center_Ki as cenKi
import matplotlib.pyplot as plt
import scipy.integrate as sp
import numpy as np
import random
from openpyxl import Workbook
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

#ダイナミクスのパラメータ
#m = 0
m1 = 17.40
m2 = 4.80
l1 = 0
l2 = 136
l3 = 14
l1g = 0
l2g = 68 
l3g = 70
Ixx_1 = 0
Iyy_1 = 0
Izz_1 = 0.35
Ixx_2 = 0.130
Iyy_2 = 0.524
Izz_2 = 0.539
Ixx_3 = 0.66
Iyy_3 = 0.0125
Izz_3 = 0.086
g = 9.8

#慣性行列
def M11(x2,x3):
    return Izz_1+Izz_2+Izz_3+m2*(l2*np.cos(x2)+l3g*np.cos(x2+x3))**2+m1*(l2g**2)*(np.cos(x2))**2
def M22(x3):
    return Iyy_2+Iyy_3+m2*(l2**2+l3g**2+2*l2*l3g*np.cos(x3)+m1*l2g**2)
def M23(x3):
    return Iyy_3+m2*(l3g**2+l2*l3g*np.cos(x3))
def M32(x3):
    return Iyy_3+m2*(l3g**2+l2*l3g*np.cos(x3))
def M33():
    return Iyy_3+m2*l3g**2

#慣性行列
def M(x2,x3):
    X = np.array([[M11(x2,x3),   0,         0   ],
                  [    0,       M22(x3),   M23(x3)],
                  [    0,       M32(x3),    M33() ]])
    return X
#重力項
def G2(x2,x3):
    return m2*(l2*g*np.cos(x2)+l3g*g*np.cos(x2+x3)+m1*l2g*g*np.cos(x2))
def G3(x2,x3):
    return m2*l3g*g*np.cos(x2+x3)

#重力項
def G(x2,x3):
    X = np.array([[    0    ],
                  [G2(x2,x3)],
                  [G3(x2,x3)]])
    #return np.round(X,8)
    return X

#コリオリ力
def H21(x2,x3):
    return m2*((l2**2)*np.sin(x2)*np.cos(x2)+(l3g**2)*np.sin(x2+x3)*np.cos(x2+x3)+l2*l3g*np.sin(2*x2+x3))+m1*(l2g**2)*np.sin(x2)*np.cos(x2)
def H23(x3):
    return -m2*l2*l3g*np.sin(x3)
def H31(x2,x3):
    return m2*(l2*l3g+np.sin(x2+x3)*np.cos(x2)+(l3g**2)*np.sin(x2+x3)*np.cos(x2+x3))
def H32(x3):
    return m2*l2*l3g*np.sin(x3)

#コリオリ力
def H(x2,x3):
    X = np.array([[     0,        0,       0   ],
                  [H21(x2,x3),    0,    H23(x3)],
                  [H31(x2,x3), H32(x3),   0    ]])
    return X

#コリオリ力：dq^2
def dq_2(x4,x5,x6):
    X = np.array([[x4**2],
                  [x5**2],
                  [x6**2]])
    return X

#遠心力
def N12_1(x2,x3):
    return -2*m2*l2*l3g*(np.sin(x2)*np.cos(x2+x3)+np.cos(x2)*np.sin(x2+x3))-2*m2*(l2**2)*np.cos(x2)*np.sin(x2)-2*m2*(l3g**2)*np.cos(x2+x3)*np.sin(x2+x3)-2*m1*(l2g**2)*np.cos(x2)*np.sin(x2)
def N13_1(x2,x3):
    return -2*m2*l2*l3g*np.cos(x2)*np.sin(x2+x3)-2*m2*(l3g**2)*np.cos(x2+x3)*np.sin(x2+x3)
def N23_2(x3):
    return -2*m2*l2*l3g*np.sin(x3)

#遠心力
def N(x2,x3):
    X = np.array([[N12_1(x2,x3), N13_1(x2,x3),   0      ],
                  [     0,            0,       N23_2(x3)],
                  [     0,            0,         0      ]])
    return X

#遠心力:dqdq
def dqdq(x4,x5,x6):
    X = np.array([[x4*x5],
                  [x4*x6],
                  [x5*x6]])
    return X

#コリオリ力＋遠心力＋重力項
def Ξ(x2,x3,x4,x5,x6):
    X = np.dot(N(x2,x3),dqdq(x4,x5,x6)) + np.dot(H(x2,x3),dq_2(x4,x5,x5)) + G(x2,x3)
    return X

def Π(x1,x2,x3,x4,x5,x6):
    X = np.array([[ 1.5*x4+0.5*np.sin(3*x1)+1.2*np.sin(x4) ],
                  [2.3*x5-1.2*np.sin(2*x2)+0.95*np.sin(x5) ],
                  [-2.1*x6-1.6*np.sin(3*x3)+0.75*np.sin(x6)]])
    return X


#制御器のパラメータ
λ = 1.4
p = 9
q = 7
k1 = 10
k2 = 5
c = 0.5
a = 0.1
β = 1.5
Kp0 = 4
Ki0  = 2
Kp = 4
Ki = 2
ρ = 0.001
ɤ = 1000

#TDFのパラメータ
u1_TDF = 0
u2_TDF = 0
u3_TDF = 0
Λ1_rec = 0
Λ2_rec = 0
Λ3_rec = 0


#ファジィ推論
def fuzzy_out_Kp(e,de):
    X = np.array([[Kp0 + β*cenKp.decision_center_Kp(e[0],de[0])],
                  [Kp0 + β*cenKp.decision_center_Kp(e[1],de[1])],
                  [Kp0 + β*cenKp.decision_center_Kp(e[2],de[2])]])
    #return np.round(X,8)
    return X
def fuzzy_out_Ki(e,de):
    X = np.array([[Kp0 + β*cenKi.decision_center_Ki(e[0],de[0])],
                  [Kp0 + β*cenKi.decision_center_Ki(e[1],de[1])],
                  [Kp0 + β*cenKi.decision_center_Ki(e[2],de[2])]])
    #return np.round(X,8)
    return X


#追従軌道
def x1d(t):
    X = np.cos((t/(5*np.pi)))-1
    return X
def x2d(t):
    X = np.cos((t/(5*np.pi))+(np.pi/2))
    return X
def x3d(t):
    X = np.sin((t/(5*np.pi))+(np.pi/2))-1
    return X

def dx1d(t):
    X = -(1/(5*np.pi))*np.sin(t/(5*np.pi))
    return X
def dx2d(t):
    X = -(1/(5*np.pi))*np.sin((t/(5*np.pi))+(np.pi/2))
    return X
def dx3d(t):
    X = (1/(5*np.pi))*np.cos((t/(5*np.pi))+(np.pi/2))
    return X
#誤差
def e(x1,x2,x3,t):
    X = np.array([[x1 - x1d(t)],
                  [x2 - x2d(t)],
                  [x3 - x3d(t)]])
    return X
def de(x4,x5,x6,t):
    X = np.array([[x4 - dx1d(t)],
                  [x5 - dx2d(t)],
                  [x6 - dx3d(t)]])
    return X
#e^[Ψ]
def e_(x1,x2,x3,t,phi):
    X = np.array([[abs(x1 - x1d(t))**phi*np.sign(x1 - x1d(t))],
                  [abs(x2 - x2d(t))**phi*np.sign(x2 - x2d(t))],
                  [abs(x3 - x3d(t))**phi*np.sign(x3 - x3d(t))]])
    return X
#de^[p/q]
def de_(x4,x5,x6,t,phi):
    X = np.array([[abs(x4 - dx1d(t))**phi*np.sign(x4 - dx1d(t))],
                  [abs(x5 - dx2d(t))**phi*np.sign(x5 - dx2d(t))],
                  [abs(x6 - dx3d(t))**phi*np.sign(x6 - dx3d(t))]])
    return X
#|e|^(λ-1)
def diag_e(x1,x2,x3,t,phi):
    X = np.array([[abs(x1 - x1d(t))**phi,           0             ,            0           ],
                  [          0            ,  abs(x2 - x2d(t))**phi,            0           ],
                  [          0            ,           0             , abs(x3 - x3d(t))**phi]])
    return X
#|de|^((p/q)-1)
def diag_de(x4,x5,x6,t,phi):
    X = np.array([[abs(x4 - dx1d(t))**(phi),                0             ,                  0           ],
                  [               0            ,  abs(x5 - dx2d(t))**(phi),                  0           ],
                  [               0            ,                0             , abs(x6 - dx3d(t))**(phi)]])
    return X


#ddxd
def ddxd(t):
    X = np.array([[       -(1/(5*np.pi))**2*np.cos(t/(5*np.pi))     ],
                  [-(1/(5*np.pi))**2*np.cos((t/(5*np.pi))+(np.pi/2))],
                  [-(1/(5*np.pi))**2*np.sin((t/(5*np.pi))+(np.pi/2))]])
    return X

#NFTSMスライディング平面
def s(x1,x2,x3,x4,x5,x6,t):
    X = e(x1,x2,x3,t) + k1*e_(x1,x2,x3,t,λ) + k2*de_(x4,x5,x6,t,p/q)
    return X


#既知関数
#f(x1,x2)
def f(x2,x3,x4,x5,x6):
    X = np.dot(np.linalg.inv(M(x2,x3)),-Ξ(x2,x3,x4,x5,x6))
    return X

#Γ
def Γ(x1,x2,x3,x4,x5,x6,s1_integ,s2_integ,s3_integ,t):
    X = Kp*s(x1,x2,x3,x4,x5,x6,t) + Ki*integ_s(s1_integ,s2_integ,s3_integ) + de(x4,x5,x6,t) + k1*λ*np.dot(diag_e(x1,x2,x3,t,(λ-1)),de(x4,x5,x6,t)) + k2*(p/q)*np.dot(diag_de(x4,x5,x6,t,((p/q)-1)),f(x2,x3,x4,x5,x6)-ddxd(t))
    return X

#Ω()
def Ω(x2,x3,x4,x5,x6,t):
    X = k2*(p/q)*np.dot(diag_de(x4,x5,x6,t,((p/p)-1)),np.linalg.inv(M(x2,x3)))
    return X

#Ω＋
def Ωp(x2,x3,x4,x5,x6,t):
    X = np.dot(np.linalg.inv(np.dot(Ω(x2,x3,x4,x5,x6,t).T,Ω(x2,x3,x4,x5,x6,t))),Ω(x2,x3,x4,x5,x6,t).T)
    return X

#未知関数
#Δ
def Δ(x1,x2,x3,x4,x5,x6,t):
    X = np.dot(np.linalg.inv(M(x2,x3)),-Π(x1,x2,x3,x4,x5,x6))
    return X

#Λ:未知関数
def Λ(x1,x2,x3,x4,x5,x6,t):
    #X = k2*(p/q)*np.dot(diag_de(x4,x5,x6,t,((p/q)-1)),Δ(x1,x2,x3,x4,x5,x6,t))
    X = np.array([[0],
                  [0],
                  [0]])    
    return X


#PID-NFTSMスライディング平面
def s_PID(x1,x2,x3,x4,x5,x6,s1_integ,s2_integ,s3_integ,u_r1,u_r2,u_r3,t):
    X = Γ(x1,x2,x3,x4,x5,x6,s1_integ,s2_integ,s3_integ,t) + np.dot(Ω(x2,x3,x4,x5,x6,t),u_PID(x1,x2,x3,x4,x5,x6,s1_integ,s2_integ,s3_integ,u_r1,u_r2,u_r3,t)) + Λ(x1,x2,x3,x4,x5,x6,t)
    return X

#∫s
def integ_s(s1_integ,s2_integ,s3_integ):
    X = np.array([[s1_integ],
                  [s2_integ],
                  [s3_integ]])
    return X


#入力関係
def u_eq(x1,x2,x3,x4,x5,x6,s1_integ,s2_integ,s3_integ,t):
    X = Γ(x1,x2,x3,x4,x5,x6,s1_integ,s2_integ,s3_integ,t)
    return X

def du_r(x1,x2,x3,x4,x5,x6,s1_integ,s2_integ,s3_integ,u_r1,u_r2,u_r3,t):
    X = (ɤ+a)*np.sign(s_PID(x1,x2,x3,x4,x5,x6,s1_integ,s2_integ,s3_integ,u_r1,u_r2,u_r3,t))
    return X

def u_r(u_r1,u_r2,u_r3):
    X = np.array([[u_r1],
                  [u_r2],
                  [u_r3]])
    return X

def u_PID(x1,x2,x3,x4,x5,x6,s1_integ,s2_integ,s3_integ,u_r1,u_r2,u_r3,t):
    X = -np.dot(Ωp(x2,x3,x4,x5,x6,t),(u_eq(x1,x2,x3,x4,x5,x6,s1_integ,s2_integ,s3_integ,t)+u_r(u_r1,u_r2,u_r3)))
    return X

def system(t, x):
    
    # 戻り値用のリスト
    y = [0]*12
    # 常微分方程式（状態方程式）の計算
   
    dx1 = x[3]
    dx2 = x[4]
    dx3 = x[5]
    dx4 = float((f(x[1],x[2],x[3],x[4],x[5]) + Δ(x[0],x[1],x[2],x[3],x[4],x[5],t) + np.dot(np.linalg.inv(M(x[1],x[2])),u_PID(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11],t)))[0][0])
    dx5 = float((f(x[1],x[2],x[3],x[4],x[5]) + Δ(x[0],x[1],x[2],x[3],x[4],x[5],t) + np.dot(np.linalg.inv(M(x[1],x[2])),u_PID(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11],t)))[0][0])
    dx6 = float((f(x[1],x[2],x[3],x[4],x[5]) + Δ(x[0],x[1],x[2],x[3],x[4],x[5],t) + np.dot(np.linalg.inv(M(x[1],x[2])),u_PID(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11],t)))[0][0])
    
    #積分値の計算
    ds1_PID = s_PID(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11],t)[0][0]
    ds2_PID = s_PID(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11],t)[1][0]
    ds3_PID = s_PID(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11],t)[2][0]

    dur1 = du_r(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11],t)[0][0]
    dur2 = du_r(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11],t)[1][0]
    dur3 = du_r(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11],t)[2][0]

    # 計算結果を返す
    y[0] = dx1
    y[1] = dx2
    y[2] = dx3
    y[3] = dx4
    y[4] = dx5
    y[5] = dx6
    y[6] = ds1_PID
    y[7] = ds2_PID
    y[8] = ds3_PID
    y[9] = dur1
    y[10] = dur2
    y[11] = dur3   

    
    return y

def simulation(x0, end, step):
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    x5 = []
    x6 = []
    t = []
    
    ode =  sp.ode(system)
    ode.set_integrator('dopri5', method='bdf')#, atol=1.e-8)
    ode.set_initial_value(x0, 0)
    t.append(0)
    x1.append(0)
    x2.append(0)
    x3.append(0)
    x4.append(0)
    x5.append(0)
    x6.append(0)

    while ode.successful() and ode.t < end - step:
        ode.integrate(ode.t + step)

        x1.append(ode.y[0]-x1d(ode.t))
        x2.append(ode.y[1]-x2d(ode.t))
        x3.append(ode.y[2]-x3d(ode.t))
        x4.append(ode.y[3])
        x5.append(ode.y[4])
        x6.append(ode.y[5])
        t.append(ode.t)
    return x1, x2, x3, x4, x5, x6, t 


F1 = np.zeros((1,6))#6行1列のゼロ行列の生成
F2 = np.zeros((1,3))
x0 = [0,0,0,0,0,0, 0,0,0, 0,0,0 ] # 初期値
end = 30        # シミュレーション時間
step = 0.001     # 時間の刻み幅

# シミュレーション
x1, x2, x3, x4, x5, x6, t  = simulation(x0, end, step)
fig, ax = plt.subplots(3, 1, figsize=(10,6))



ax[0].grid()

ax[0].plot(t, x1)

ax[1].grid()

ax[1].plot(t, x2)

ax[2].grid()

ax[2].plot(t, x3)

plt.show()