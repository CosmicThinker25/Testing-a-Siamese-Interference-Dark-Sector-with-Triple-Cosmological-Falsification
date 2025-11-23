"""
phase_field_KG_solver_f1_0_L4_0_65_v3.py
Integración de la ecuación de Klein–Gordon para el campo de fase PNGB
en un fondo cosmológico FLRW con radiación + materia + energía de fase.

Produce un CSV con:
a, N, phi, dphi_dN, Delta_phi, H, rho_r, rho_m, rho_DE, Omega_DE,
rho_tot, p_tot, w_eff, w_phi
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint

# ================================================================
#  PARÁMETROS DEL MODELO
# ================================================================
F = 1.0         # escala de simetría f — NUEVO
LAMBDA4 = 0.65  # densidad del potencial PNGB hoy
OMEGA_R0 = 9e-5 # radiación de Planck 2018

# Condiciones iniciales a = 1 (N = ln a = 0)
PHI_1 = 0.2      # fase presente
DPHI_DN_1 = 0.0  # campo detenido hoy
Y0 = [PHI_1, DPHI_DN_1]

# Rango de integración en N = ln a (hacia atrás)
N_start = 0.0           # a = 1
N_end = -5.0            # a ≈ exp(-5) ≈ 0.0067 (suficiente)
N_points = 2000
N_array = np.linspace(N_start, N_end, N_points)


# ================================================================
# FUNCIONES DEL MODELO
# ================================================================
def V(phi):
    """Potencial PNGB"""
    return LAMBDA4 * (1.0 - np.cos(phi / F))

def dV_dphi(phi):
    """Derivada del potencial PNGB"""
    return LAMBDA4 * (1.0 / F) * np.sin(phi / F)


def system_equations(y, N):
    """Sistema de EDO para (phi, dphi/dN)"""
    phi, dphi_dN = y
    a = np.exp(N)

    # Densidades físicas
    rho_r = OMEGA_R0 * a**(-4)
    rho_m = (1.0 - LAMBDA4 - OMEGA_R0) * a**(-3)  # emergente
    dotphi_sq = (dphi_dN / a)**2 / 2.0            # 0.5 * (dphi/dt)^2
    rho_DE = dotphi_sq + V(phi)
    rho_tot = rho_r + rho_m + rho_DE
    H = np.sqrt(rho_tot)

    # Presión
    p_phi = dotphi_sq - V(phi)
    p_tot = p_phi + rho_r / 3.0
    w_eff = p_tot / rho_tot

    # Ecuaciones de movimiento
    dphi_dN_dt = dphi_dN
    d_dphi_dN_dt = -(3.0 * (1.0 + w_eff) - 1.0) * dphi_dN - (1.0 / H**2) * dV_dphi(phi)

    return [dphi_dN_dt, d_dphi_dN_dt]


# ================================================================
# INTEGRACIÓN
# ================================================================
sol = odeint(system_equations, Y0, N_array)
phi_vals = sol[:,0]
dphi_dN_vals = sol[:,1]

# ================================================================
# COMPILACIÓN DE RESULTADOS
# ================================================================
rows = []
for N, phi, dphi_dN in zip(N_array, phi_vals, dphi_dN_vals):
    a = np.exp(N)
    rho_r = OMEGA_R0 * a**(-4)
    rho_m = (1.0 - LAMBDA4 - OMEGA_R0) * a**(-3)
    dotphi_sq = (dphi_dN / a)**2 / 2.0
    rho_DE = dotphi_sq + V(phi)
    rho_tot = rho_r + rho_m + rho_DE
    H = np.sqrt(rho_tot)

    p_phi = dotphi_sq - V(phi)
    p_tot = p_phi + rho_r / 3.0
    w_eff = p_tot / rho_tot
    w_phi = p_phi / rho_DE

    rows.append([
        a, N, phi, dphi_dN, 2 * phi, H,
        rho_r, rho_m, rho_DE, rho_DE / rho_tot,
        rho_tot, p_tot, w_eff, w_phi
    ])

df = pd.DataFrame(rows, columns=[
    "a", "N", "phi", "dphi_dN", "Delta_phi", "H",
    "rho_r", "rho_m", "rho_DE", "Omega_DE",
    "rho_tot", "p_tot", "w_eff", "w_phi"
])

# ================================================================
# GUARDADO DEL CSV
# ================================================================
filename = f"../results/csv/KG_PNGB_f1_0_L4_0_65_phi0_0_2_v3.csv"
df.to_csv(filename, index=False)

print("\nIntegración completada correctamente.")
print(f"CSV guardado en:\n   {filename}\n")
print("Columnas incluidas:")
print(", ".join(df.columns))
