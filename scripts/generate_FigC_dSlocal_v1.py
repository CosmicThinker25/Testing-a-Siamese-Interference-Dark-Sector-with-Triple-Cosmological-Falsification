#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
generate_FigC_dSlocal_v1.py

Figura C — Tasa de interferencia siamesa:
S_local(N) y dS_local/dN como función de N = ln a.

Usa el CSV físico "bueno":
  ../results/csv/KG_PNGB_fixed_phiFreeze_f1_2_phi0_3_0_v2.csv
generado por phase_field_KG_solver_fixed_phiFreeze_v2.

La figura se guarda en:
  ../results/figures_v2/FigC_dSlocal_v1.png
  ../results/figures_v2/FigC_dSlocal_v1.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1. Rutas y lectura del CSV
# ---------------------------------------------------------------------

CSV_PATH = "../results/csv/KG_PNGB_fixed_phiFreeze_f1_2_phi0_3_0_v2.csv"
OUT_DIR = "../results/figures_v2"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"📂 Leyendo datos desde: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

# Esperamos como mínimo estas columnas:
#   a, N, Delta_phi
# Si ya existieran columnas S_local o dS_local, las ignoramos y
# las recomputamos de forma consistente a partir de Delta_phi.
required_cols = ["a", "N", "Delta_phi"]
for col in required_cols:
    if col not in df.columns:
        raise RuntimeError(f"Columna requerida '{col}' no encontrada en el CSV.")

a = df["a"].values
N = df["N"].values
Delta_phi = df["Delta_phi"].values

# ---------------------------------------------------------------------
# 2. Definición de S_local y dS_local/dN
# ---------------------------------------------------------------------
# Definimos una entropía siamés local normalizada a partir del desfase:
#   S_local = (1 - cos(Delta_phi)) / 2
# Esta elección es:
#   - acotada entre 0 y 1,
#   - máxima para desfase antípoda,
#   - cero para sincronía perfecta (Delta_phi = 0).
#
# Luego calculamos su derivada respecto a N = ln a con np.gradient.

S_local = 0.5 * (1.0 - np.cos(Delta_phi))

# Derivada numérica dS/dN
dS_dN = np.gradient(S_local, N)

# Opcional: pequeño suavizado para que la figura sea más legible
# (ventana de media móvil). Si no quieres suavizado, comenta este bloque.
window = 9  # número impar
if window > 1:
    kernel = np.ones(window) / window
    S_local_smooth = np.convolve(S_local, kernel, mode="same")
    dS_dN_smooth = np.convolve(dS_dN, kernel, mode="same")
else:
    S_local_smooth = S_local
    dS_dN_smooth = dS_dN

# ---------------------------------------------------------------------
# 3. Figura C: S_local y dS_local/dN vs N
# ---------------------------------------------------------------------

fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

# --- Panel superior: S_local(N) ---
ax[0].plot(N, S_local_smooth, label=r"$S_{\rm local}(N)$", linewidth=2.0)
ax[0].set_ylabel(r"$S_{\rm local}$ (normalizada)", fontsize=12)
ax[0].set_title(
    r"Figura C — Dinámica de la interferencia siamés: "
    r"$S_{\rm local}(N)$ y ${\rm d}S_{\rm local}/{\rm d}N$",
    fontsize=14
)
ax[0].grid(True, alpha=0.3)
ax[0].legend(loc="upper left", fontsize=11)

# Marcamos el presente N=0
ax[0].axvline(0.0, color="k", linestyle=":", alpha=0.6)
ax[0].text(0.01, 0.05, "Hoy", transform=ax[0].transAxes,
           fontsize=10, verticalalignment="bottom")

# --- Panel inferior: dS_local/dN(N) ---
ax[1].plot(N, dS_dN_smooth, label=r"${\rm d}S_{\rm local}/{\rm d}N$", linewidth=2.0)
ax[1].axhline(0.0, color="k", linestyle="--", alpha=0.5)

ax[1].set_xlabel(r"$N = \ln a$", fontsize=12)
ax[1].set_ylabel(r"${\rm d}S_{\rm local}/{\rm d}N$", fontsize=12)
ax[1].grid(True, alpha=0.3)
ax[1].legend(loc="upper left", fontsize=11)

# Podemos resaltar la ventana donde la tasa de interferencia es máxima.
# Buscamos el máximo en |dS_dN| y lo marcamos.
idx_peak = np.argmax(np.abs(dS_dN_smooth))
N_peak = N[idx_peak]
dS_peak = dS_dN_smooth[idx_peak]

ax[1].axvline(N_peak, color="r", linestyle=":", alpha=0.7)
ax[1].annotate(
    r"máxima tasa de interferencia",
    xy=(N_peak, dS_peak),
    xytext=(N_peak + 0.5, dS_peak * 0.5),
    arrowprops=dict(arrowstyle="->", alpha=0.7),
    fontsize=10
)

plt.tight_layout()

png_path = os.path.join(OUT_DIR, "FigC_dSlocal_v1.png")
pdf_path = os.path.join(OUT_DIR, "FigC_dSlocal_v1.pdf")
fig.savefig(png_path, dpi=200)
fig.savefig(pdf_path)

print("✅ Figura C generada.")
print(f"PNG: {png_path}")
print(f"PDF: {pdf_path}")
plt.close(fig)
