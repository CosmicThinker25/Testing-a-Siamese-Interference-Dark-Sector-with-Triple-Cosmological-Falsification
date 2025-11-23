import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

########################################
# CONFIGURACIÓN
########################################
CSV_SIAMESE = "../results/csv/KG_PNGB_fixed_phiFreeze_f1_2_phi0_3_0_v2_csv"
OUTPUT_DIR = "../results/figures_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Datos BAO + CC (conjunto S, H/H0)
BAO_CC_DATA = np.array([
    [0.07, 1.02, 0.08],
    [0.12, 1.09, 0.10],
    [0.17, 1.15, 0.09],
    [0.24, 1.22, 0.11],
    [0.34, 1.30, 0.12],
    [0.43, 1.37, 0.14],
    [0.57, 1.45, 0.12],
    [0.73, 1.52, 0.16],
    [0.90, 1.62, 0.15],
    [1.30, 1.78, 0.18],
    [1.50, 1.86, 0.20],
    [2.33, 2.05, 0.26],
])

########################################
# CARGAR CURVA DEL MODELO SIAMÉS
########################################
df = pd.read_csv(CSV_SIAMESE)

a = df["a"].values
H = df["H"].values

# Convertir a redshift
z = 1/a - 1

# Ordenar (por si el CSV tiene pasos invertidos)
sort_idx = np.argsort(z)
z = z[sort_idx]
H = H[sort_idx]

########################################
# CURVA ΛCDM ESTÁNDAR H/H0
########################################
Omega_m = 0.30
Omega_r = 9e-5
Omega_L = 1 - Omega_m - Omega_r

H_LCDM = np.sqrt(Omega_r * (1+z)**4 + Omega_m * (1+z)**3 + Omega_L)

########################################
# FIGURA A (DOS PANELES)
########################################
fig, axes = plt.subplots(2, 1, figsize=(9, 11), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1]})

##########################
# PANEL SUPERIOR — COMPLETO
##########################
axes[0].plot(z, H, color="red", label="Modelo PNGB Siamés", linewidth=2)
axes[0].plot(z, H_LCDM, "k--", label="ΛCDM (Planck)", linewidth=2)
axes[0].errorbar(BAO_CC_DATA[:,0], BAO_CC_DATA[:,1],
                 yerr=BAO_CC_DATA[:,2], fmt="o", color="tab:blue",
                 label="BAO + CC (conjunto S)")

axes[0].set_ylabel("H(z)  (unidades: H₀ = 1)", fontsize=12)
axes[0].set_title("Figura A — Historia de Expansión: Modelo Siamés vs ΛCDM + BAO/CC", fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.25)

##########################
# PANEL INFERIOR — ZOOM
##########################
mask = z <= 2.0
axes[1].plot(z[mask], H[mask], color="red", linewidth=2)
axes[1].plot(z[mask], H_LCDM[mask], "k--", linewidth=2)
axes[1].errorbar(BAO_CC_DATA[:,0], BAO_CC_DATA[:,1],
                 yerr=BAO_CC_DATA[:,2], fmt="o", color="tab:blue")

axes[1].set_ylim(0.7, 2.2)         # ← ZOOM IDEAL
axes[1].set_xlim(0, 3.0)
axes[1].set_ylabel("H/H₀ (zoom)", fontsize=12)
axes[1].set_xlabel("Redshift  z", fontsize=13)
axes[1].grid(alpha=0.25)

########################################
# GUARDAR
########################################
plt.tight_layout()
png_path = os.path.join(OUTPUT_DIR, "FigA_Hz_two_panels_v3.png")
pdf_path = os.path.join(OUTPUT_DIR, "FigA_Hz_two_panels_v3.pdf")
plt.savefig(png_path, dpi=300)
plt.savefig(pdf_path)
plt.close()

print("\n🎉 FIGURA A GENERADA EXITOSAMENTE")
print("PNG guardado en:", png_path)
print("PDF guardado en:", pdf_path)
