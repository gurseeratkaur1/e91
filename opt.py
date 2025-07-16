import numpy as np
import matplotlib.pyplot as plt

# -------------------- Parameters for Optical Fibre --------------------
eta_det = 0.6                 # Detector efficiency
d = 4e-8                      # Dark count probability per detector
c = 0.01                      # Intrinsic error rate (from alignment, source noise)
alpha_db_per_km = 0.2         # Fibre attenuation in dB/km
alpha = alpha_db_per_km / 10  # Convert to base-10 exponent
p_straycounts = 0             # Assume negligible straycounts in fibre
f_e = 1.16                    # Error correction inefficiency factor

# -------------------- Channel Length --------------------
L_km = np.linspace(0.1, 100, 500)  # from 0.1 to 100 km

# -------------------- Fibre Transmittance --------------------
eta_T = 10 ** (-alpha * L_km)      # channel transmittance
alpha_L = eta_det * eta_T          # combined with detector efficiency
alpha_x = np.sqrt(alpha_L)         # assuming source in center

# -------------------- Coincidence Probabilities --------------------
p_true = eta_det * alpha_L
p_false = 8 * alpha_x * d + 16 * d**2
p_coin = p_true + p_false + p_straycounts

# -------------------- QBER Calculation --------------------
numerator = c * p_true + 0.5 * (p_false + p_straycounts)
e_bbm92 = numerator / p_coin

# -------------------- Binary Entropy Function --------------------
def binary_entropy(x):
    x = np.clip(x, 1e-12, 1 - 1e-12)  # to avoid log(0)
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

# -------------------- Secret Key Rate (SKR) --------------------
skr = 0.5 * p_coin * ( 1- f_e * binary_entropy(e_bbm92))

# -------------------- Plot 1: QBER --------------------
plt.figure(figsize=(10, 5))
plt.plot(L_km, e_bbm92, color='tab:blue', label='QBER')
plt.xlabel("Fibre Length (km)")
plt.ylabel("Quantum Bit Error Rate (QBER)")
plt.title("BBM92 QBER vs Fibre Length (Optical Fibre Channel)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------- Plot 2: SKR --------------------
plt.figure(figsize=(10, 5))
plt.plot(L_km, skr, color='tab:green', label='Secret Key Rate (SKR)')
plt.xlabel("Fibre Length (km)")
plt.ylabel("Secret Key Rate")
plt.title("BBM92 SKR vs Fibre Length (Optical Fibre Channel)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
