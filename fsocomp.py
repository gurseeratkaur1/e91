import numpy as np
import matplotlib.pyplot as plt

# Common Constants ===================
eta_det = 0.6              # Detector efficiency
d = 8e-6                   # Dark count probability
c = 0.01                   # Intrinsic error rate
alpha_db = 0.5             # Atmospheric attenuation (dB/km)
alpha = alpha_db / (4.343*1000)   # Convert dB/km to 1/m
d_t = 0.2                  # Transmitter aperture diameter (m)
d_r = 0.2                  # Receiver aperture diameter (m)
D = 0.3e-3                 # Beam divergence (radians)
f_e_bbm92 = 1.16           # Error correction inefficiency factor (BBM92)
f_ec_bb84 = 1.16           # Error correction inefficiency factor (BB84)

# Distance values ==================

L_km = np.linspace(0.1, 50, 500)
L_m = L_km * 1000

# BBM92 Calculations ===================

geometric_loss = (d_r / (d_t + D * L_m))**2
atmospheric_loss = np.exp(-alpha * L_m)
eta_T_bbm92 = geometric_loss * atmospheric_loss
alpha_L = eta_det * eta_T_bbm92
alpha_x = np.sqrt(alpha_L)

p_true = eta_det * alpha_L
p_false = 8 * alpha_x * d + 16 * d**2
p_straycounts = 5e-6
p_coin = p_true + p_false + p_straycounts

e_bbm92 = (c * p_true + 0.5 * (p_false + p_straycounts)) / p_coin

def tau_bbm92(e):
    val = 0.5 + 2 * e - 2 * e**2
    val = np.clip(val, 1e-12, 1)  # avoid log2(0) or negative
    return np.where(e < 0.5, -np.log2(val), 0)


def binary_entropy(x):
    x = np.clip(x, 1e-12, 1 - 1e-12)
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

skr_bbm92 = p_coin * 0.5 * (tau_bbm92(e_bbm92) - f_e_bbm92 * binary_entropy(e_bbm92))

# BB84 Calculations ===================

mu = 0.5
c_error = 0.02

def eta_t(L):  # L in meters
    denom = d_t + D * L
    geom = (d_r / denom) ** 2
    atmos = np.exp(-alpha * L)
    return geom * atmos

def p_stray():
    return 5e-6

def p_dark():
    return 4e-8

def p_signal_bb84(eta):
    return 1 - np.exp(-eta_det * eta * mu)

def p_prime(mu):
    return 1 - (1 + mu + mu*2 / 2 + mu*3 / 12) * np.exp(-mu)

def tau_bb84(e):
    return -1*np.log2(0.5 + 2 * e - 2 * e**2) if e < 0.5 else 0

def f_e(e): return f_ec_bb84

def QBER_BB84(eta):
    signal = p_signal_bb84(eta)
    stray = p_stray()
    dark = p_dark()
    return (c_error * signal + 0.5 * (dark + stray)) / (signal + dark + stray)

def beta(p_click_val):
    p_p = p_prime(mu)
    result = (p_click_val - p_p) / p_click_val
    return np.clip(result, 1e-6, 1)

def R_BB84(eta):
    signal = p_signal_bb84(eta)
    stray = p_stray()
    dark = p_dark()
    p_click = signal + dark + stray
    e = QBER_BB84(eta)
    e = np.clip(e, 1e-12, 1 - 1e-12)
    
    beta_val = beta(p_click)
    e_div_beta = np.clip(e / beta_val, 1e-12, 1 - 1e-12)
    tau_val = np.vectorize(tau_bb84)(e_div_beta)
    
    h_e = -e * np.log2(e) - (1 - e) * np.log2(1 - e)
    skr = 0.5 * p_click * (beta_val * tau_val - f_e(e) * h_e)
    skr[e > 0.11] = 0
    skr[skr < 0] = 0
    return skr

eta_vals_bb84 = eta_t(L_m)
qber_bb84 = QBER_BB84(eta_vals_bb84)
skr_bb84 = R_BB84(eta_vals_bb84)

# Plot QBER ===================

plt.figure(figsize=(10, 5))
plt.plot(L_km, e_bbm92, label='QBER (BBM92)', color='tab:blue')
plt.plot(L_km, qber_bb84, label='QBER (BB84)', color='tab:purple', linestyle='--')
plt.xlabel("FSO Link Distance (km)")
plt.ylabel("Quantum Bit Error Rate (QBER)")
plt.title("QBER vs Distance for BB84 and BBM92 (FSO)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot SKR ===================

plt.figure(figsize=(10, 5))
plt.plot(L_km, skr_bbm92, label='SKR (BBM92)', color='tab:green')
plt.plot(L_km, skr_bb84, label='SKR (BB84)', color='tab:orange', linestyle='--')
plt.xlabel("FSO Link Distance (km)")
plt.ylabel("Secret Key Rate (bits/pulse)")
plt.title("Secret Key Rate vs Distance for BB84 and BBM92 (FSO)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
