import numpy as np
import matplotlib.pyplot as plt

# Constants
eta_det = 0.6              # Detector efficiency
d = 4e-8                   # Dark count probability per detector
c = 0.01                   # Intrinsic error rate
alpha = 0.5 / 4.343        # Atmospheric attenuation coefficient (from dB/km to 1/km)
d_t = 0.2                  # Transmitter aperture diameter (m)
d_r = 0.2                  # Receiver aperture diameter (m)
D = 0.5e-3                 # Beam divergence (radians)
f_e = 1.16                 # Error correction inefficiency factor

# Range of channel lengths (in km and meters)
L_km = np.linspace(0.1, 50, 200)
L_m = L_km * 1000

# Free-space transmittance: geometric + atmospheric
geometric_loss = (d_r / (d_t + D * L_m))**2
atmospheric_loss = np.exp(-alpha * L_km)
eta_T = geometric_loss * atmospheric_loss

# Combined channel and detector efficiency
alpha_L = eta_det * eta_T

# Assume source at center: alpha_x = alpha_{L-x} = sqrt(alpha_L)
alpha_x = np.sqrt(alpha_L)

# Coincidence probabilities
p_true = eta_det * alpha_L
p_false = 8 * alpha_x * d + 16 * d**2
p_straycounts = 5e-6  # stray environmental photons
p_coin = p_true + p_false + p_straycounts

# QBER for BBM92
numerator = c * p_true + 0.5 * (p_false + p_straycounts)
e_bbm92 = numerator / p_coin

# Binary entropy function
def binary_entropy(x):
    x = np.clip(x, 1e-12, 1 - 1e-12)
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

# SKR calculation
skr = p_coin * 0.5 * (f_e*binary_entropy(e_bbm92))

# ---------------------- Plot 1: QBER ----------------------
plt.figure(figsize=(10, 5))
plt.plot(L_km, e_bbm92, color='tab:blue', label='QBER')
plt.xlabel("Channel Length (km)")
plt.ylabel("Quantum Bit Error Rate (QBER)")
plt.title("BBM92 QBER vs Channel Length (FSO with Stray Counts)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------- Plot 2: SKR ----------------------
plt.figure(figsize=(10, 5))
plt.plot(L_km, skr, color='tab:green', label='Secret Key Rate')
plt.xlabel("Channel Length (km)")
plt.ylabel("Secret Key Rate (SKR)")
plt.title("BBM92 SKR vs Channel Length (FSO with Stray Counts)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()