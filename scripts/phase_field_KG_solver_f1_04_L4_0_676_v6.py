import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# ======================================================================
#  PNGB Siamese phase field solver in cosmological background
#  Version: v6 — f = 1.04, Λ⁴ = 0.676 (ajuste fino)
#  Output:  ../results/csv/KG_PNGB_f1_04_L4_0_676_phi0_0_2_v6.csv
# ======================================================================

# ----- Parámetros cosmológicos (H0 = 1, rho_crit0 = 1) -----

LAMBDA4  = 0.676        # ajuste fino de la fracción de energía oscura
F        = 1.04         # escala PNGB calibrada
OMEGA_R0 = 9e-5         # radiación hoy

# Universo plano => Omega_m0 sale de cierre
OMEGA_M0 = 1.0 - LAMBDA4 - OMEGA_R0


# ----- Condiciones iniciales en a = 1 -----

phi_today     = 0.2      # valor del campo hoy
dphi_dN_today = 0.0      # campo casi detenido hoy

# Dominio de integración en N = ln(a)
N_min  = np.log(1e-2)     # a = 0.01 (z ≈ 99)
N_max  = 0.0
N_eval = np.linspace(N_min, N_max, 500)


# ======================================================================
#  Potencial PNGB y fondo cosmológico
# ======================================================================

def V(phi):
    """Potencial PNGB."""
    return LAMBDA4 * (1.0 - np.cos(phi / F))


def dV_dphi(phi):
    """Derivada del potencial PNGB."""
    return LAMBDA4 * (1.0 / F) * np.sin(phi / F)


def compute_background(N, phi, dphi_dN):
    """
    Devuelve a, H, rho_r, rho_m, rho_DE, rho_tot para un N y estado dados.
    """
    a = np.exp(N)

    # Fluidos estándar
    rho_r = OMEGA_R0 * a**(-4)
    rho_m = OMEGA_M0 * a**(-3)

    # Primera estimación de H (sirve para estimar la cinética)
    H_sq = rho_r + rho_m + LAMBDA4
    H = np.sqrt(max(H_sq, 1e-16))

    # Refinar con término cinético del campo
    dot_phi = H * dphi_dN
    rho_kin = 0.5 * dot_phi**2
    rho_DE = rho_kin + V(phi)
    rho_tot = rho_r + rho_m + rho_DE
    H = np.sqrt(max(rho_tot, 1e-16))

    return a, H, rho_r, rho_m, rho_DE, rho_tot


# ======================================================================
#  Sistema de EDO en N = ln(a)
# ======================================================================

def rhs(N, y):
    """
    Sistema en N:
        y[0] = phi
        y[1] = dphi/dN
    """
    phi, dphi_dN = y

    a, H, rho_r, rho_m, rho_DE, rho_tot = compute_background(N, phi, dphi_dN)

    # Presiones
    dot_phi = H * dphi_dN
    rho_kin = 0.5 * dot_phi**2
    p_phi = rho_kin - V(phi)
    p_r = rho_r / 3.0
    p_tot = p_r + p_phi

    w_eff = p_tot / rho_tot if rho_tot > 0 else -1.0
    dlnH_dN = -1.5 * (1.0 + w_eff)  # de ecuaciones de Friedmann + continuidad

    dphi_dN_eq = dphi_dN
    d2phi_dN2 = - (3.0 + dlnH_dN) * dphi_dN - dV_dphi(phi) / (H**2)

    return [dphi_dN_eq, d2phi_dN2]


# ======================================================================
#  Ejecución principal
# ======================================================================

def main():
    print("\nIntegrando ecuación de fase PNGB (v6, ajuste fino)...")
    print(f"   Parámetros: f = {F}, Λ⁴ = {LAMBDA4}, Ω_r0 = {OMEGA_R0:.1e}")
    print(f"   Ω_m0 (emergente) = {OMEGA_M0:.6f}\n")

    sol = solve_ivp(
        rhs,
        (N_min, N_max),
        [phi_today, dphi_dN_today],
        t_eval=N_eval,
        atol=1e-9,
        rtol=1e-7
    )

    if not sol.success:
        print("⚠️  Error en la integración:", sol.message)
        return

    phi_vals = sol.y[0]
    dphi_dN_vals = sol.y[1]

    rows = []
    for N, phi, dphi_dN in zip(N_eval, phi_vals, dphi_dN_vals):
        a, H, rho_r, rho_m, rho_DE, rho_tot = compute_background(N, phi, dphi_dN)
        dot_phi = H * dphi_dN
        rho_kin = 0.5 * dot_phi**2
        p_phi = rho_kin - V(phi)
        p_r = rho_r / 3.0
        p_tot = p_r + p_phi
        w_eff = p_tot / rho_tot if rho_tot > 0 else -1.0
        w_phi = p_phi / rho_DE if rho_DE > 0 else -1.0
        Delta_phi = 2.0 * phi

        rows.append([
            a, N, phi, dphi_dN, Delta_phi, H,
            rho_r, rho_m, rho_DE, rho_tot, p_tot, w_eff, w_phi
        ])

    df = pd.DataFrame(rows, columns=[
        "a", "N", "phi", "dphi_dN", "Delta_phi", "H",
        "rho_r", "rho_m", "rho_DE", "rho_tot", "p_tot", "w_eff", "w_phi"
    ])

    df = df.sort_values("a").reset_index(drop=True)

    out_csv = "../results/csv/KG_PNGB_f1_04_L4_0_676_phi0_0_2_v6.csv"
    df.to_csv(out_csv, index=False)

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
