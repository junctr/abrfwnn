import csv
import numpy as np
from matplotlib import pyplot as plt
"""
e_0.append(e[0][0])

e_3.append(tau[0])
e_6.append(taud[0])
e_9.append(y[0])
e_12.append(taus0[0])
e_15.append(K@s[0])
e_18.append(taus1[0])
e_21.append(wn[0])

e_24.append(e[0][1])
e_27.append(s[0])
e_30.append(D[0][0])

e_33.append(beta.T @ omega)
"""
n_seed = 0
alpha_lambda = 0.3
alpha_wn0 = 100
alpha_wn1 = 10

alpha_0s0 = 5.0
alpha_0s1 = 5.0
alpha_0s2 = 5.0

alpha_1s0 = 5.0
alpha_1s1 = 5.0
alpha_1s2 = 5.0

T = 1000
step = 0.0001
end = 100

end_plt = 100
start_plt = 0

n_e = 3

t_data = np.loadtxt(f"abrfwnn/data/step{step}_t{end}.csv")

e_all_p = np.loadtxt(f"abrfwnn/data_test_old_d/p_s{n_seed}_m{alpha_lambda}_wn{alpha_wn0}_{alpha_wn1}_s{alpha_0s0}_{alpha_0s1}_{alpha_0s2}_T{T}_step{step}_t{end}_e_all.csv")
# e_all_c = np.loadtxt(f"abrfwnn/data_test_seed/p_s{n_seed}_m{alpha_lambda}_wn{alpha_wn0}_{alpha_wn1}_s{alpha_1s0}_{alpha_1s1}_{alpha_1s2}_T{T}_step{step}_t{end}_e_all.csv")

fig, axes = plt.subplots(nrows=11, ncols=3, sharex=False)

for i in range(11):
    
    for j in range(3):
        # axes[i,j].plot(t_data, e_all_c[3*i+j])
        axes[i,j].plot(t_data, e_all_p[3*i+j])
        
        # axes[i,j].plot(t_data, e_all_c[3*i+j], color="tab:green", label = "Conventional")
        # axes[i,j].plot(t_data, e_all_p[3*i+j], color="tab:red", label = "Proposed")
        # axes[i,j].legend()
        
        axes[i,j].grid()

# plt.savefig(f"abrfwnn/data_test/s{n_seed}_m{alpha_lambda}_wn{alpha_wn0}_{alpha_wn1}_s{alpha_s0}_{alpha_s1}_{alpha_s2}_T{T}_step{step}_t{end}_all.png")

plt.show()
