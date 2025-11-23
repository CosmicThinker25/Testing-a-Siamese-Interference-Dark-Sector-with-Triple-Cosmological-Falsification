import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURACIÓN BÁSICA
# ==============================================================================

# Ruta del CSV generado por phase_field_KG_solver.py
DATA_FILE = os.path.join("..", "results", "csv", "KG_PNGB_f0_8_L4_0_7_phi0_0_2_csv")

# Carpeta de salida para las figuras
FIG_DIR = os.path.join("..", "results", "figures_v2")
os.makedirs(FIG_DIR, exist_ok=True)

# Cargar datos
df = pd.read_csv(DATA_FILE)

# Extraer columnas básicas
a = df["a"].values
N = df["N"].values
phi = df["phi"].values
dphi_dN = df["dphi_dN"].values
Delta_phi = df["Delta_phi"].values
H = df["H"].values
rho_r = df["rho_r"].values
rho_m = df["rho_m"].values
rho_DE = df["rho_DE"].values
Omega_DE = df["Omega_DE"].values
rho_tot = df["rho_tot"].values
p_tot = df["p_tot"].values
w_eff = df["w_eff"].values
w_phi = df["w_phi"].values

# Variable redshift
z = 1.0 / a - 1.0

# Densidad de materia oscura efectiva (en este toy model la identificamos con rho_m)
rho_DM = rho_m.copy()
rho_eff = rho_tot.copy()

# ==============================================================================
# MODELO ΛCDM DE REFERENCIA PARA H(z)
# ==============================================================================

# Definimos un ΛCDM "de referencia" estándar (no tiene que coincidir exactamente con nuestro toy model)
OMEGA_R_LCDM = 9.0e-5
OMEGA_M_LCDM = 0.3
OMEGA_L_LCDM = 1.0 - OMEGA_R_LCDM - OMEGA_M_LCDM

# Tomamos H0 de nuestro modelo en a=1 para que ambas curvas coincidan hoy
H0_model = H[np.argmax(a)]  # último valor, a=1
H_LCDM = H0_model * np.sqrt(
    OMEGA_R_LCDM * a**(-4.0) + OMEGA_M_LCDM * a**(-3.0) + OMEGA_L_LCDM
)

# ==============================================================================
# FIGURA 1: Componentes de densidad vs a (ρ_r, ρ_DM, ρ_DE, ρ_tot)
# ==============================================================================

plt.figure(figsize=(7, 5))
plt.loglog(a, rho_r, label=r"$\rho_r(a)$ (radiation)")
plt.loglog(a, rho_DM, label=r"$\rho_{\mathrm{DM}}(a)$ (effective)")
plt.loglog(a, rho_DE, label=r"$\rho_{\mathrm{DE}}(a)$ (phase field)")
plt.loglog(a, rho_eff, label=r"$\rho_{\mathrm{tot}}(a)$", linestyle="--")

plt.xlabel(r"Scale factor $a$")
plt.ylabel(r"Density (arb. units)")
plt.title("Density components vs scale factor")
plt.legend()
plt.tight_layout()
fig1_path = os.path.join(FIG_DIR, "Figure1_rho_components_v2.png")
plt.savefig(fig1_path, dpi=300)
plt.close()

# ==============================================================================
# FIGURA 2: H(a) y comparación con ΛCDM
# ==============================================================================

plt.figure(figsize=(7, 5))
plt.loglog(a, H, label="Siamese PNGB model")
plt.loglog(a, H_LCDM, label=r"$\Lambda$CDM reference", linestyle="--")

plt.xlabel(r"Scale factor $a$")
plt.ylabel(r"$H(a)$ (arb. units)")
plt.title("Expansion history: Siamese PNGB vs $\Lambda$CDM")
plt.legend()
plt.tight_layout()
fig2_path = os.path.join(FIG_DIR, "Figure2_H_vs_LCDM_v2.png")
plt.savefig(fig2_path, dpi=300)
plt.close()

# También podemos hacer H(z) en escala lineal para z<10
mask_z = z < 10.0
z_small = z[mask_z]
H_small = H[mask_z]
H_LCDM_small = H_LCDM[mask_z]

plt.figure(figsize=(7, 5))
plt.plot(z_small, H_small, label="Siamese PNGB model")
plt.plot(z_small, H_LCDM_small, label=r"$\Lambda$CDM reference", linestyle="--")
plt.gca().invert_xaxis()  # para que z decrezca hacia la derecha

plt.xlabel(r"Redshift $z$")
plt.ylabel(r"$H(z)$ (arb. units)")
plt.title("Expansion history (zoom): $H(z)$ for $z<10$")
plt.legend()
plt.tight_layout()
fig2b_path = os.path.join(FIG_DIR, "Figure2b_Hz_zoom_v2.png")
plt.savefig(fig2b_path, dpi=300)
plt.close()

# ==============================================================================
# FIGURA 3: Ecuaciones de estado w_phi(a) y w_eff(a)
# ==============================================================================

plt.figure(figsize=(7, 5))
plt.plot(a, w_phi, label=r"$w_{\phi}(a)$ (phase field)")
plt.plot(a, w_eff, label=r"$w_{\mathrm{eff}}(a)$ (total)", linestyle="--")

plt.xlabel(r"Scale factor $a$")
plt.ylabel(r"Equation of state $w$")
plt.title("Equation of state: field and effective")
plt.axhline(-1.0, linestyle=":", label=r"$w=-1$")
plt.axhline(0.0, linestyle=":", label=r"$w=0$")
plt.axhline(1.0, linestyle=":", label=r"$w=1$")
plt.legend()
plt.tight_layout()
fig3_path = os.path.join(FIG_DIR, "Figure3_w_phi_w_eff_v2.png")
plt.savefig(fig3_path, dpi=300)
plt.close()

# ==============================================================================
# FIGURA 4: Fase Δφ(a)
# ==============================================================================

plt.figure(figsize=(7, 5))
plt.plot(a, Delta_phi)

plt.xlabel(r"Scale factor $a$")
plt.ylabel(r"$\Delta\phi(a)$")
plt.title("Siamese phase drift $\Delta\phi(a)$")
plt.tight_layout()
fig4_path = os.path.join(FIG_DIR, "Figure4_DeltaPhi_v2.png")
plt.savefig(fig4_path, dpi=300)
plt.close()

# ==============================================================================
# FIGURA 5: Entropía Siamés (normalizada) y su derivada
# ==============================================================================

# Definimos una entropía local S_local(a) ∝ desincronización de fase normalizada
Delta_min = Delta_phi.min()
Delta_max = Delta_phi.max()
Delta_span = Delta_max - Delta_min if Delta_max != Delta_min else 1.0

S_local = (Delta_phi - Delta_min) / Delta_span  # normalizada a [0,1]
ln_a = np.log(a)
dS_dln_a = np.gradient(S_local, ln_a)

# S_local(a)
plt.figure(figsize=(7, 5))
plt.plot(a, S_local)
plt.xlabel(r"Scale factor $a$")
plt.ylabel(r"$S_{\mathrm{local}}(a)$ (normalized)")
plt.title("Siamese entropy proxy $S_{\mathrm{local}}(\Delta\phi(a))$")
plt.tight_layout()
fig5_path = os.path.join(FIG_DIR, "Figure5_S_local_v2.png")
plt.savefig(fig5_path, dpi=300)
plt.close()

# dS/d ln a
plt.figure(figsize=(7, 5))
plt.plot(a, dS_dln_a)
plt.xlabel(r"Scale factor $a$")
plt.ylabel(r"$dS_{\mathrm{local}}/d\ln a$")
plt.title("Entropy growth rate $dS_{\mathrm{local}}/d\ln a$")
plt.axhline(0.0, linestyle="--")
plt.tight_layout()
fig5b_path = os.path.join(FIG_DIR, "Figure5b_dS_dln_a_v2.png")
plt.savefig(fig5b_path, dpi=300)
plt.close()

# ==============================================================================
# FIGURA 6 (OPCIONAL): Omega_DE(a)
# ==============================================================================

plt.figure(figsize=(7, 5))
plt.semilogx(a, Omega_DE)
plt.xlabel(r"Scale factor $a$")
plt.ylabel(r"$\Omega_{\mathrm{DE}}(a)$")
plt.title("Dark-energy fraction $\Omega_{\mathrm{DE}}(a)$ in the Siamese PNGB model")
plt.tight_layout()
fig6_path = os.path.join(FIG_DIR, "Figure6_OmegaDE_v2.png")
plt.savefig(fig6_path, dpi=300)
plt.close()

# ==============================================================================
# RESUMEN
# ==============================================================================

print("✅ Figuras generadas en:", FIG_DIR)
print("   -", os.path.basename(fig1_path))
print("   -", os.path.basename(fig2_path))
print("   -", os.path.basename(fig2b_path))
print("   -", os.path.basename(fig3_path))
print("   -", os.path.basename(fig4_path))
print("   -", os.path.basename(fig5_path))
print("   -", os.path.basename(fig5b_path))
print("   -", os.path.basename(fig6_path))
