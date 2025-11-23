#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FIGURA B — w_eff(z) (Modelo Siamés)
----------------------------------
Genera la figura directamente desde el CSV original
sin modificar datos ni aplicar ajustes cosmológicos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === Archivo de datos fijo ===
CSV = "../results/csv/KG_PNGB_fixed_phiFreeze_f1_2_phi0_3_0_v2.csv"

# === Lectura ===
df = pd.read_csv(CSV)
a = df["a"].values
z = (1/a) - 1
w_eff = df["w_eff"].values
w_phi = df["w_phi"].values

# === Rango de interés para la figura ===
mask = (z >= 0) & (z <= 4)
z = z[mask]
w_eff = w_eff[mask]
w_phi = w_phi[mask]

# === Curva ΛCDM: w = -1 ± 0.1 (banda) ===
lcdm_center = -1
lcdm_low = -1.1
lcdm_high = -0.9

# === Figura ===
plt.figure(figsize=(9, 6))

# Banda gris
plt.fill_between(z, lcdm_low, lcdm_high, color="lightgray",
                 alpha=0.7, label="ΛCDM (w = -1 ± 0.1)")

# Curvas teóricas del modelo Siamés
plt.plot(z, w_eff, color="blue", linewidth=2.5,
         label=r"$w_{\mathrm{eff}}(z)$ (observable)")
plt.plot(z, w_phi, color="orange", linestyle="--", linewidth=2,
         label=r"$w_{\phi}(z)$ (campo fundamental)")

plt.axhline(-1, color="black", linestyle=":", linewidth=1.5)

plt.xlabel("Redshift  z", fontsize=13)
plt.ylabel("Parámetro de estado  w(z)", fontsize=13)
plt.ylim(-1.4, 0.4)
plt.xlim(0, 4)

plt.title("Figura B — Parámetro de estado de la energía oscura", fontsize=15)
plt.legend(fontsize=11)

os.makedirs("../results/figures_v2/", exist_ok=True)
outfile = "../results/figures_v2/FigB_w_eff_v2.pdf"
plt.savefig(outfile, dpi=300, bbox_inches="tight")

print("\n🎉 FIGURA GENERADA EXITOSAMENTE")
print(f"📁 Guardada en: {outfile}\n")
