import numpy as np
from scipy.integrate import odeint
import pandas as pd

# ================================================================
#  PNGB Siamese Phase Solver — phi frozen today (FIXED v2)
# ================================================================

# === Parámetros cosmológicos base ===
OMEGA_R0 = 9e-5
OMEGA_M0 = 0.350        # Materia observada (no se ajusta a mano)
OMEGA_DE0 = 1.0 - OMEGA_R0 - OMEGA_M0

# === Parámetros del campo de fase ===
F = 1.2                  # Escala de simetría f
phi_today = 3.0          # *** CAMBIO CRÍTICO → campo casi congelado ***
dphi_dN_today = 0.0      # φ_N hoy = 0 para w ≈ -1

# === Calibración automática de Λ⁴ para que V(φ_today) = Ω_DE0 ===
Lambda4 = OMEGA_DE0 / (1.0 - np.cos(phi_today / F))
print(f"Λ^4 calibrada = {Lambda4:.6f}")

# === Rango de integración ===
N_ini = 0.0        # ln(a=1) → presente
N_min = -7.0       # ln(a ≈ 0.001) → temprano
N_points = 2000
N_array = np.linspace(N_ini, N_min, N_points)

# ------------------------------------------------
#  POTENCIAL PNGB
# ------------------------------------------------
def V(phi):
    return Lambda4 * (1 - np.cos(phi / F))

def V_prime(phi):
    return Lambda4 * (1.0 / F) * np.sin(phi / F)

# ------------------------------------------------
#  SISTEMA DE ODES EN VARIABLE N = ln(a)
# ------------------------------------------------
def system(y, N):
    phi, phi_N = y
    a = np.exp(N)

    # Energía cinética y densidades
    rho_r = OMEGA_R0 * a**(-4)
    rho_m = OMEGA_M0 * a**(-3)
    rho_phi = 0.5 * (phi_N**2) + V(phi)

    rho_tot = rho_r + rho_m + rho_phi
    H = np.sqrt(rho_tot)

    # Presiones
    p_phi = 0.5 * (phi_N**2) - V(phi)
    p_tot = p_phi + (1/3) * rho_r

    w_eff = p_tot / rho_tot

    # Ecuación de Klein–Gordon en N
    dphi_dN = phi_N
    dphiN_dN = - (3 - 1.5 * (1 + w_eff)) * phi_N - V_prime(phi) / (H**2)

    return [dphi_dN, dphiN_dN]


# === Integración ===
y0 = [phi_today, dphi_dN_today]
sol = odeint(system, y0, N_array)

phi_vals = sol[:, 0]
phiN_vals = sol[:, 1]

# === Construcción de columnas derivadas ===
rows = []
for N, phi, phi_N in zip(N_array, phi_vals, phiN_vals):
    a = np.exp(N)
    rho_r = OMEGA_R0 * a**(-4)
    rho_m = OMEGA_M0 * a**(-3)
    rho_phi = 0.5 * (phi_N**2) + V(phi)
    rho_tot = rho_r + rho_m + rho_phi
    p_phi = 0.5 * (phi_N**2) - V(phi)
    p_tot = p_phi + (1/3) * rho_r
    w_eff = p_tot / rho_tot
    w_phi = p_phi / rho_phi
    H = np.sqrt(rho_tot)
    Delta_phi = 2 * phi

    rows.append([a, N, phi, phi_N, Delta_phi, H,
                 rho_r, rho_m, rho_phi, rho_tot, p_phi, p_tot, w_eff, w_phi])

df = pd.DataFrame(rows, columns=[
    'a', 'N', 'phi', 'dphi_dN', 'Delta_phi', 'H',
    'rho_r', 'rho_m', 'rho_phi', 'rho_tot', 'p_phi', 'p_tot',
    'w_eff', 'w_phi'
])

fname = f"KG_PNGB_fixed_phiFreeze_f{F}_phi0_{phi_today}_v2.csv".replace('.', '_')
df.to_csv(f"../results/csv/{fname}", index=False)

print("\n=== FINAL ===")
print(f"H(a=1)  = {df['H'].iloc[0]:.6f}")
print(f"w_phi(1)= {df['w_phi'].iloc[0]:.6f}")
print(f"w_eff(1)= {df['w_eff'].iloc[0]:.6f}")
print(f"CSV guardado en ../results/csv/{fname}")
