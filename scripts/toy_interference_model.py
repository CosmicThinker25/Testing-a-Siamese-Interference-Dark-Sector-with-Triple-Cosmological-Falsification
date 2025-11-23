import numpy as np
import pandas as pd

N = 2000
a = np.linspace(0.05, 1, N)  # empezamos en a=0.05 para evitar divergencias visuales

# Densidades reescaladas para que ρ+ = 1 en a = 1
rho_plus = (a ** -3)
rho_plus /= rho_plus[-1]

rho_minus = 0.85 * (a ** -3)
rho_minus /= rho_plus[-1]  # se mantiene relativa a rho_plus en a=1

# Desincronización de fase Δφ(a)
Delta_phi = 0.15 + 1.3 * (1 - np.exp(-5 * (1 - a)))  # suave, creciente

# Interferencia
rho_eff = rho_plus + rho_minus + 2 * np.sqrt(rho_plus * rho_minus) * np.cos(Delta_phi)

rho_DM = rho_plus + rho_minus
rho_DE = rho_eff - rho_DM

df = pd.DataFrame({
    "a": a,
    "Delta_phi": Delta_phi,
    "rho_plus": rho_plus,
    "rho_minus": rho_minus,
    "rho_eff": rho_eff,
    "rho_DM": rho_DM,
    "rho_DE": rho_DE,
})

df.to_csv("../results/csv/toy_interference_output.csv", index=False)

print("✔ CSV regenerado sin divergencias iniciales.")
