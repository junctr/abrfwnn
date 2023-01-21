import numpy as np
import time
#import scipy.integrate as sp
from matplotlib import pyplot as plt
from tqdm import tqdm
import csv

step = 0.0001

alpha_d = 10 * np.identity(3, dtype=np.float64)
alpha_dk = 0.2 * np.identity(3, dtype=np.float64)
alpha_lambda = 0.3

s = 0.5 * np.ones((3,1), dtype=np.float64)

D = np.array([
        [1.0,0.0,0.0],
        [0.0,1.0,0.0],
        [0.0,0.0,1.0]],
        dtype=np.float64
    )


D += step * (alpha_d @ np.ones((3,1), dtype=np.float64) @ s.T - alpha_d @ alpha_dk @ D * np.linalg.norm(s))
# D += step * (alpha_d @ np.ones((3,1), dtype=np.float64) @ s.T - alpha_d @ alpha_dk @ D * np.linalg.norm(s)) + alpha_lambda * (Dold[-1] - Dold[-2])


print(np.ones((3,1), dtype=np.float64) @ s.T)
print(D * np.linalg.norm(s))
print(alpha_d @ np.ones((3,1), dtype=np.float64) @ s.T)
print( - alpha_d @ alpha_dk @ D * np.linalg.norm(s))
print(np.round(D,4))