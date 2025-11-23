import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# ================================================================
#  PNGB Siamese phase field in cosmological background
#  Version: v4  (f = 1.2, LAMBDA4 = 0.65)
#  Output CSV: ../results/csv/KG_PNGB_f1_2_L4_0_65_phi0_0_2_v4.csv
# ================================================================

# ----- Cosmological parameters (H0 = 1, rho_crit0 = 1 units) -----

LAMBDA4 = 0.65          # ~ Omega_DE,0
F       = 1.2           # PNGB scale f
OMEGA_R0 = 9e-5         # radiation today

# Flat universe: Omega_m0 emerges from closure
OMEGA_M0 = 1.0 - LAMBDA4 - OMEGA_R0

# ----- Initial conditions at a = 1 (today) -----

phi_today      = 0.2    # phase field value at a = 1
dphi_dN_today  = 0.0    # field at rest today in e-folds N = ln a

# Integration range in N = ln a (from N_min to 0)
N_min = np.log(1e-2)    # a_min = 1e-2  (z ~ 99)
N_max = 0.0
num_points = 500


# ================================================================
#  Helper functions
# ================================================================

def potential(phi):
    """
    PNGB potential:
        V(phi) = LAMBDA4 * [1 - cos(phi / F)]
    """
    return LAMBDA4 * (1.0 - np.cos(phi / F))


def dV_dphi(phi):
    """
    Derivative of PNGB potential: V'(phi)
    """
    return LAMBDA4 * (1.0 / F) * np.sin(phi / F)


def rho_components(N, phi, dphi_dN):
    """
    Compute rho_r, rho_m, rho_DE, rho_tot and H for given N, phi, dphi/dN.
    Units: H0 = 1 => rho_crit0 = 1.
    """
    a = np.exp(N)

    # Standard components
    rho_r = OMEGA_R0 * a**(-4.0)
    rho_m = OMEGA_M0 * a**(-3.0)

    # Phase field kinetic term: dot(phi) = H * dphi/dN
    # We don't yet know H, so we'll solve self-consistently below.
    # Start with a guess H^2 ~ rho_r + rho_m + LAMBDA4 and refine.

    # First rough guess for H^2 (good enough because field is subdominant early)
    H2_guess = rho_r + rho_m + LAMBDA4
    H_guess = np.sqrt(H2_guess)

    # With that H, compute kinetic energy and refine:
    dot_phi = H_guess * dphi_dN
    rho_kin = 0.5 * dot_phi**2
    rho_DE = rho_kin + potential(phi)

    rho_tot = rho_r + rho_m + rho_DE
    H = np.sqrt(rho_tot)

    # Recompute with updated H (one iteration is enough for consistency)
    dot_phi = H * dphi_dN
    rho_kin = 0.5 * dot_phi**2
    rho_DE = rho_kin + potential(phi)
    rho_tot = rho_r + rho_m + rho_DE
    H = np.sqrt(rho_tot)

    return a, H, rho_r, rho_m, rho_DE, rho_tot


def rhs(N, y):
    """
    System of ODEs in terms of N = ln a:
        y[0] = phi
        y[1] = dphi/dN
    """
    phi, dphi_dN = y

    # Get background quantities
    a, H, rho_r, rho_m, rho_DE, rho_tot = rho_components(N, phi, dphi_dN)

    # Total pressure: p_r = rho_r/3, p_m = 0, p_phi = rho_kin - V
    # First compute kinetic and potential
    dot_phi = H * dphi_dN
    rho_kin = 0.5 * dot_phi**2
    V_phi = potential(phi)
    p_phi = rho_kin - V_phi

    p_tot = (1.0 / 3.0) * rho_r + p_phi
    w_eff = p_tot / rho_tot

    # dphi/dN = dphi_dN
    dphi_dN_eq = dphi_dN

    # d^2 phi / dN^2
    # Klein–Gordon in N:
    #   d^2 phi/dN^2 + (3 + d ln H / dN) dphi/dN + (1/H^2) dV/dphi = 0
    # with:
    #   d ln H / dN = - (3/2) (1 + w_eff)
    dlnH_dN = -1.5 * (1.0 + w_eff)
    friction = 3.0 + dlnH_dN
    d2phi_dN2 = - friction * dphi_dN - dV_dphi(phi) / (H**2)

    return [dphi_dN_eq, d2phi_dN2]


# ================================================================
#  Integration
# ================================================================

def main():
    N_vals = np.linspace(N_min, N_max, num_points)

    y0 = [phi_today, dphi_dN_today]

    print("Integrando ecuación de fase PNGB en fondo cosmológico (N = ln a)...")
    print(f"   Parámetros: F = {F}, LAMBDA4 = {LAMBDA4}, OMEGA_R0 = {OMEGA_R0:.1e}")
    print(f"   OMEGA_M0 (emergente) = {OMEGA_M0:.6f}")

    sol = solve_ivp(
        rhs,
        t_span=(N_min, N_max),
        y0=y0,
        t_eval=N_vals,
        rtol=1e-7,
        atol=1e-9,
        dense_output=False
    )

    if not sol.success:
        print("⚠️  Integración NO exitosa:", sol.message)
        return

    phi_vals = sol.y[0]
    dphi_dN_vals = sol.y[1]

    # Compute derived quantities
    a_list = []
    H_list = []
    rho_r_list = []
    rho_m_list = []
    rho_DE_list = []
    rho_tot_list = []
    p_tot_list = []
    w_eff_list = []
    w_phi_list = []
    Delta_phi_list = []

    for N, phi, dphi_dN in zip(N_vals, phi_vals, dphi_dN_vals):
        a, H, rho_r, rho_m, rho_DE, rho_tot = rho_components(N, phi, dphi_dN)

        # Kinetic, potential, pressures
        dot_phi = H * dphi_dN
        rho_kin = 0.5 * dot_phi**2
        V_phi = potential(phi)
        p_phi = rho_kin - V_phi
        p_tot = (1.0 / 3.0) * rho_r + p_phi

        w_eff = p_tot / rho_tot
        w_phi = p_phi / rho_DE if rho_DE > 0 else 0.0

        Delta_phi = 2.0 * phi  # siamese phase difference

        a_list.append(a)
        H_list.append(H)
        rho_r_list.append(rho_r)
        rho_m_list.append(rho_m)
        rho_DE_list.append(rho_DE)
        rho_tot_list.append(rho_tot)
        p_tot_list.append(p_tot)
        w_eff_list.append(w_eff)
        w_phi_list.append(w_phi)
        Delta_phi_list.append(Delta_phi)

    # Put in DataFrame
    df = pd.DataFrame({
        "a": a_list,
        "N": N_vals,
        "phi": phi_vals,
        "dphi_dN": dphi_dN_vals,
        "Delta_phi": Delta_phi_list,
        "H": H_list,
        "rho_r": rho_r_list,
        "rho_m": rho_m_list,
        "rho_DE": rho_DE_list,
        "rho_tot": rho_tot_list,
        "p_tot": p_tot_list,
        "w_eff": w_eff_list,
        "w_phi": w_phi_list,
    })

    # Sort by scale factor ascending
    df = df.sort_values("a").reset_index(drop=True)

    # Save CSV
    outname = "../results/csv/KG_PNGB_f1_2_L4_0_65_phi0_0_2_v4.csv"
    df.to_csv(outname, index=False)

    # Print summary at a = 1 (last row)
    row_today = df.iloc[-1]
    print("✅ Integración completada correctamente.")
    print(f"   a(último)      = {row_today['a']:.6f}")
    print(f"   H(a=1)        ≈ {row_today['H']:.6f} (debería ser ~1)")
    print(f"   w_phi(a=1)    ≈ {row_today['w_phi']:.6f}")
    print(f"   w_eff(a=1)    ≈ {row_today['w_eff']:.6f}")
    print()
    print("📁 Resultados guardados en:")
    print(f"   {outname}")
    print()
    print("Columnas incluidas:")
    print("   a, N, phi, dphi_dN, Delta_phi, H, rho_r, rho_m, rho_DE,")
    print("   rho_tot, p_tot, w_eff, w_phi")


if __name__ == "__main__":
    main()
