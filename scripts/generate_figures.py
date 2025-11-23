import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../results/csv/toy_interference_output.csv")

# Plot 1 — Effective density
plt.plot(df["a"], df["rho_plus"], label="ρ+")
plt.plot(df["a"], df["rho_minus"], label="ρ−")
plt.plot(df["a"], df["rho_eff"], label="ρ_eff", linewidth=2)
plt.xlabel("a (scale factor)")
plt.ylabel("Energy density (arb.)")
plt.title("Siamese Interference Effective Energy Density")
plt.legend()
plt.savefig("../results/figures/Figure1_rho_eff.png", dpi=300)
plt.clf()

# Plot 2 — Dark Matter vs Dark Energy components
plt.plot(df["a"], df["rho_DM"], label="ρ_DM (interpretation)")
plt.plot(df["a"], df["rho_DE"], label="ρ_DE (interference surplus)")
plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("a")
plt.ylabel("Energy density (arb.)")
plt.title("Dark sector decomposition")
plt.legend()
plt.savefig("../results/figures/Figure2_DM_DE_split.png", dpi=300)
plt.clf()

# Plot 3 — Phase evolution
plt.plot(df["a"], df["Delta_phi"])
plt.xlabel("a")
plt.ylabel("Δφ (rad)")
plt.title("Phase desynchronization between Siamese Universes")
plt.savefig("../results/figures/Figure3_DeltaPhi.png", dpi=300)
plt.clf()

print("✔ Figuras generadas correctamente.")
