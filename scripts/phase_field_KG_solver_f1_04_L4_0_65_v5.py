import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# ======================================================================
#  PNGB Siamese phase field solver in cosmological background
#  Version: v5 — calibrated for f = 1.04, Λ⁴ = 0.65
#  Output:  ../results/csv/KG_PNGB_f1_04_L4_0_65_phi0_0_2_v5.csv
# ======================================================================

# ----- Cosmological parameters (H0 = 1, rho_crit0 = 1) -----

LAMBDA4  = 0.65          # Present-day DE fraction
F        = 1.04          # PNGB scale (calibrated)
OMEGA_R0 = 9e-5          # Radiation today

# Flat universe => Omega_m0 emerges from closure
OMEGA_M0 = 1.0 - LAMBDA4 - OMEGA_R0


# ----- Initial conditions at a = 1 -----

phi_today     = 0.2      # field value at a = 1
dphi_dN_today = 0.0      # field at rest today

# Integration domain in N = ln(a)
N_min  = np.log(1e-2)     # a = 0.01 (z ≈ 99)
N_max  = 0.0
N_eval = np.linspace(N_min, N_max, 500)


# ======================================================================
#  PNGB potential and background
# ======================================================================

def V(phi):
    """PNGB potential."""
    return LAMBDA4 * (1.0 - np.cos(phi / F))


def dV_dphi(phi):
    """Derivative of PNGB potential."""
    return LAMBDA4 * (1.0 / F) * np.sin(phi / F)


def compute_background(N, phi, dphi_dN):
    a = np.exp(N)

    # Standard fluids
    rho_r = OMEGA_R0 * a**(-4)
    rho_m = OMEGA_M0 * a**(-3)

    # First guess for H (used to evaluate KE)
    H_sq = rho_r + rho_m + LAMBDA4
    H = np.sqrt(H_sq)

    # Refine with kinetic term
    dot_phi = H * dphi_dN
    rho_kin = 0.5 * dot_phi**2
    rho_DE = rho_kin + V(phi)
    rho_tot = rho_r + rho_m + rho_DE
    H = np.sqrt(rho_tot)

    return a, H, rho_r, rho_m, rho_DE, rho_tot


# ======================================================================
#  ODE system in N = ln(a)
# ======================================================================

def rhs(N, y):
    phi, dphi_dN = y

    a, H, rho_r, rho_m, rho_DE, rho_tot = compute_background(N, phi, dphi_dN)

    # Pressures
    dot_phi = H * dphi_dN
    rho_kin = 0.5 * dot_phi**2
    p_phi = rho_kin - V(phi)
    p_tot = (1.0/3.0) * rho_r + p_phi

    w_eff = p_tot / rho_tot
    dlnH_dN = -1.5 * (1.0 + w_eff)    # continuity relation

    dphi_dN_eq = dphi_dN
    d2phi_dN2 = - (3.0 + dlnH_dN) * dphi_dN - dV_dphi(phi) / (H**2)

    return [dphi_dN_eq, d2phi_dN2]


# ======================================================================
#  Main execution
# ======================================================================

def main():
    print("\nIntegrando ecuación de fase PNGB (v5)...")
    print(f"   Parámetros: f = {F}, Λ⁴ = {LAMBDA4}, Ω_r0 = {OMEGA_R0:.1e}")
    print(f"   Ω_m0 (emergente) = {OMEGA_M0:.6f}\n")

    sol = solve_ivp(rhs, (N_min, N_max), [phi_today, dphi_dN_today],
                    t_eval=N_eval, atol=1e-9, rtol=1e-7)

    if not sol.success:
        print("⚠️  Error en la integración:", sol.message)
        return

    phi_vals = sol.y[0]
    dphi_dN_vals = sol.y[1]

    # Derived quantities
    rows = []
    for N, phi, dphi_dN in zip(N_eval, phi_vals, dphi_dN_vals):
        a, H, rho_r, rho_m, rho_DE, rho_tot = compute_background(N, phi, dphi_dN)

        dot_phi = H * dphi_dN
        rho_kin = 0.5 * dot_phi**2
        p_phi = rho_kin - V(phi)
        p_tot = (1.0/3.0) * rho_r + p_phi
        w_eff = p_tot / rho_tot
        w_phi = p_phi / rho_DE if rho_DE > 0 else -1

        Delta_phi = 2.0 * phi

        rows.append([a, N, phi, dphi_dN, Delta_phi, H,
                     rho_r, rho_m, rho_DE, rho_tot, p_tot, w_eff, w_phi])

    df = pd.DataFrame(rows, columns=[
        "a", "N", "phi", "dphi_dN", "Delta_phi", "H",
        "rho_r", "rho_m", "rho_DE", "rho_tot", "p_tot", "w_eff", "w_phi"
    ])

    df = df.sort_values("a").reset_index(drop=True)

    out_csv = "../results/csv/KG_PNGB_f1_04_L4_0_65_phi0_0_2_v5.csv"
    df.to_csv(out_csv, index=False)

    # Report at a = 1
    today = df.iloc[-1]
    print("✅ Integración completada.\n")
    print(f"   a(último)      = {today['a']:.6f}")
    print(f"   H(a=1)        ≈ {today['H']:.6f}")
    print(f"   w_phi(a=1)    ≈ {today['w_phi']:.6f}")
    print(f"   w_eff(a=1)    ≈ {today['w_eff']:.6f}\n")
    print("📁 Resultados guardados en:")
    print(f"   {out_csv}\n")
    print("Columnas incluidas:")
    print("   a, N, phi, dphi_dN, Delta_phi, H, rho_r, rho_m, rho_DE,")
    print("   rho_tot, p_tot, w_eff, w_phi")


if __name__ == "__main__":
    main()
