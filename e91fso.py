import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, pi, sqrt, log2, exp

# ========================== Constants ==========================
alpha_db = 0.1         # Atmospheric attenuation (dB/km)
alpha = alpha_db / (4.343)   # Convert dB/km to 1/km
D = 0.025e-3              # Beam divergence (rad)
d_t = 0.01                # Transmitter aperture diameter (m)
d_r = 0.03                # Receiver aperture diameter (m)
eta = 0.6                 # Detector efficiency
eta_c = 0.6               # Collection efficiency
eta_t = eta * eta_c       # Total detection efficiency

# === New: Separate noise components ===
P_stray = 5e-6            # Stray light noise probability
P_dark = 5e-6             # Dark count probability
P_nc = P_stray + P_dark   # Total noise count probability per detector

nu_s = 0.64e6             # Source pair generation rate (pairs per second)

# Analyzer angles for CHSH inequality (in radians)
theta_1A = 0
theta_3A = pi / 4
theta_1B = -pi / 8
theta_3B = pi / 8
phi = pi                  # Entangled state phase (maximally entangled)

# Binary entropy function
def h(x):
    epsilon = 1e-12
    x = np.clip(x, epsilon, 1 - epsilon)
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

# Transmittance calculation
def transmittance(L_km):
    L_m = L_km * 1000
    geo_loss = (d_r / (d_t + D * L_m))**2
    atm_loss = exp(-alpha * L_km)
    return geo_loss * atm_loss

# Normalization factor N
def compute_N(T):
    p_s = T**2
    p_1 = 2 * T * (1 - T)
    p_0 = (1 - T)**2
    click_prob = eta_t + 2*P_nc * (1 - eta_t)
    numerator = p_s * eta_t**2
    denominator = (p_s * click_prob**2 +
                   2*p_1 * P_nc * click_prob +
                   4*p_0 * P_nc**2)
    return numerator / denominator

# Correlation function
def E(theta_A, theta_B, N, phi):
    return N * (-cos(2*theta_A) * cos(2*theta_B) +
                cos(phi) * sin(2*theta_A) * sin(2*theta_B))

# Bell parameter S
def compute_S(N, phi):
    E1 = E(theta_1A, theta_1B, N, phi)
    E2 = E(theta_1A, theta_3B, N, phi)
    E3 = E(theta_3A, theta_1B, N, phi)
    E4 = E(theta_3A, theta_3B, N, phi)
    return abs(E1 + E2 - E3 + E4)

# QBER from S
def compute_QBER(S):
    return 0.5 * (1 - S / (2 * sqrt(2)))

# SKR using Acín et al.'s formula
def compute_SKR(S, Q, T):
    if S <= 2:
        return 0
    term = (1 + sqrt(S**2 / 4 - 1)) / 2
    return (1/3) * nu_s * T * (1 - h(Q) - h(term))

# Simulate E91 for one distance
def simulate_E91(L_km):
    T = transmittance(L_km)
    N = compute_N(T)
    S = compute_S(N, phi)
    Q = compute_QBER(S)
    SKR = compute_SKR(S, Q, T)
    return L_km, S, Q * 100, SKR

# =================== Run Simulation ===================
dist_range = np.linspace(0.01, 50, 300)
S_vals, Q_vals, SKR_vals = [], [], []

for L in dist_range:
    _, S, Q, SKR = simulate_E91(L)
    S_vals.append(S)
    Q_vals.append(Q)
    SKR_vals.append(SKR)
    SKR_vals_clipped = np.clip(SKR_vals, 0, 2e4)  # Cap at 200,000 bps

# =================== Plot 1: Bell Parameter S ===================
plt.figure(figsize=(10, 5))
plt.plot(dist_range, S_vals, label="S (CHSH)", color='blue')
plt.axhline(y=2, color='red', linestyle='--', label="S = 2 Threshold")
plt.xlabel("Distance (km)")
plt.ylabel("Bell Parameter S")
plt.title("S vs Distance (E91 - FSO Channel)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =================== Plot 2: QBER ===================
plt.figure(figsize=(10, 5))
plt.plot(dist_range, Q_vals, label="QBER (%)", color='orange')
plt.axhline(y=14.6, color='red', linestyle='--', label="QBER Threshold")
plt.xlabel("Distance (km)")
plt.ylabel("QBER (%)")
plt.title("QBER vs Distance (E91 - FSO Channel)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =================== Plot 3: SKR ===================
plt.figure(figsize=(10, 5))
plt.plot(dist_range, SKR_vals_clipped, label="SKR (bps)", color='green')
plt.xlabel("Distance (km)")
plt.ylabel("Secret Key Rate (bps)")
plt.title("SKR vs Distance (E91 - FSO Channel)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


