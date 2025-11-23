import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cargamos el resultado del modelo interferencial
df = pd.read_csv("../results/csv/toy_interference_output.csv")

a = df["a"].values
Delta_phi = df["Delta_phi"].values

# Normalizamos la desincronización para construir una entropía de juguete
# Queremos: Delta_phi ≈ 0  →  S_local ≈ 0  (estado pre-temporal)
#           Delta_phi grande → S_local grande (flecha del tiempo)
dphi_shifted = Delta_phi - Delta_phi.min()
dphi_norm = dphi_shifted / dphi_shifted.max()

S_local = dphi_norm          # entropía local adimensional
S_global = np.zeros_like(a)  # en el modelo siamés ideal S_+ + S_- ≈ 0

# Derivada aproximada para ilustrar la "flecha" (dS/dln a)
ln_a = np.log(a)
dS_dln_a = np.gradient(S_local, ln_a)

# Figura 4 — Entropía local vs escala
plt.plot(a, S_local, label="S_local(a) (normalized)")
plt.axhline(0, linewidth=0.8)
plt.xlabel("a (scale factor)")
plt.ylabel("S_local (arb. units)")
plt.title("Siamese local entropy driven by phase desynchronization")
plt.legend()
plt.savefig("../results/figures/Figure4_S_local_vs_a.png", dpi=300)
plt.clf()

# Figura 5 — Flecha del tiempo (dS/dln a)
plt.plot(a, dS_dln_a)
plt.axhline(0, linewidth=0.8)
plt.xlabel("a (scale factor)")
plt.ylabel("dS_local / d ln a")
plt.title("Emergent arrow of time from phase drift")
plt.savefig("../results/figures/Figure5_dS_dln_a.png", dpi=300)
plt.clf()

print("✔ Figuras de entropía siamesa generadas:")
print("  - Figure4_S_local_vs_a.png")
print("  - Figure5_dS_dln_a.png")
