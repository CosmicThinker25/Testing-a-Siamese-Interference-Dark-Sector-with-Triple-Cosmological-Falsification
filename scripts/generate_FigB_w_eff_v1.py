import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === INPUT CSV ===
CSV = "KG_PNGB_fixed_phiFreeze_f1_2_phi0_3_0_v2.csv"

df = pd.read_csv(CSV)

# Convertimos a z
z = (1.0 / df["a"]) - 1.0

# Variables
w_eff = df["w_eff"]
w_phi = df["w_phi"]

# Figura
plt.figure(figsize=(8, 6))

# Líneas
plt.plot(z, w_eff, label=r"$w_{\rm eff}(z)$ (Modelo Siamés)", color="red", linewidth=2.3)
plt.plot(z, w_phi, label=r"$w_{\phi}(z)$ (Campo PNGB)", color="orange", linewidth=2.0)

# Línea ΛCDM
plt.axhline(-1, color="black", linestyle="--", linewidth=1.4, label=r"$w = -1$ (ΛCDM)")

# Banda observacional
plt.fill_between(z, -1.1, -0.9, color="gray", alpha=0.25, label="Rango Observacional Típico")

# Estética
plt.xlim(0, 5)
plt.ylim(-2.0, 0.0)
plt.xlabel(r"Redshift  $z$")
plt.ylabel(r"Ecuación de Estado  $w$")
plt.title(r"Figura B — $w(z)$ del Modelo Siamés vs. Valor $\Lambda$CDM")
plt.grid(True, alpha=0.25)
plt.legend(fontsize=9)
plt.tight_layout()

# Guardado
out = "../results/figures_v2/FigB_w_eff_v1.png"
plt.savefig(out, dpi=300)
print(f"🖼 Figura generada en: {out}")
