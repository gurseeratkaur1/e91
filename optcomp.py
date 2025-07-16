import numpy as np
import matplotlib.pyplot as plt

# -------------------- Fibre Length --------------------
L_km = np.linspace(0.1, 100, 500)
L_m = L_km * 1000

# -------------------- Common Parameters --------------------
eta_det = 0.6
alpha_db_per_km = 0.2
alpha = alpha_db_per_km / 10  # for base-10 log
eta_T = 10 ** (-alpha * L_km)

# -------------------- BBM92 QBER --------------------
d_bbm = 4e-8
c_bbm = 0.01
p_stray_bbm = 0

alpha_L = eta_det * eta_T
alpha_x = np.sqrt(alpha_L)

p_true = eta_det * alpha_L
p_false = 8 * alpha_x * d_bbm + 16 * d_bbm**2
p_coin = p_true + p_false + p_stray_bbm
numerator_bbm = c_bbm * p_true + 0.5 * (p_false + p_stray_bbm)
qber_bbm92 = numerator_bbm / p_coin

# -------------------- BB84 QBER --------------------
mu = 0.5
D = 4e-8
t_w = 0.5e-9
d_bb84 = D * t_w
c_error_bb84 = 0.02

def p_signal_bb84(eta):
    return 1 - np.exp(-eta_det * eta * mu)

def QBER_BB84(eta):
    signal = p_signal_bb84(eta)
    dark = d_bb84
    return (c_error_bb84 * signal + 0.5 * dark) / (signal + dark)

qber_bb84 = QBER_BB84(eta_T)

# -------------------- Plot --------------------
plt.figure(figsize=(10, 5))
# plt.plot(L_km, qber_bbm92, label='BBM92 (Fibre)', color='tab:blue')
# plt.plot(L_km, qber_bb84, label='BB84 (Fibre)', color='tab:orange', linestyle='--')
plt.semilogy(L_km, qber_bbm92, label='BBM92 (Fibre)', color='tab:blue')
plt.semilogy(L_km, qber_bb84, label='BB84 (Fibre)', color='tab:orange', linestyle='--')

plt.xlabel("Fibre Length (km)")
plt.ylabel("Quantum Bit Error Rate (QBER)")
plt.title("QBER vs Fibre Length: BB84 vs BBM92 (Optical Fibre)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
