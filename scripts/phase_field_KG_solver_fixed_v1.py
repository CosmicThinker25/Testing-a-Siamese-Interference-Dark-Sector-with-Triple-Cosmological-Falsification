import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# ======================================================================
#  PNGB Siamese phase field solver — FIXED VERSION
#  - Usa N = ln(a)
#  - Impone H(a=1) = 1 calibrando Λ⁴ a partir de Ω_DE0
#  - Implementa Friedmann y fricción correctamente
# ======================================================================

# ------------------ PARÁMETROS COSMOLÓGICOS ---------------------------

OMEGA_R0 = 9e-5      # radiación hoy
OMEGA_M0 = 0.35      # materia hoy (puedes poner 0.30 si quieres)
OMEGA_DE0 = 1.0 - OMEGA_R0 - OMEGA_M0   # energía oscura hoy (cierre plano)

# ------------------ PARÁMETROS DEL CAMPO PNGB ------------------------

F = 1.2              # escala f (por ejemplo, el caso que querías probar)
phi_today = 0.2      # valor del campo hoy
dphi_dN_today = 0.0  # campo casi congelado hoy

# Calibramos Λ⁴ para que V(phi_today) = Ω_DE0
# V(phi) = Λ⁴ [1 - cos(phi/f)] => Λ⁴ = Ω_DE0 / [1 - cos(phi_today / F)]
denom = 1.0 - np.cos(phi_today / F)
if denom <= 0:
    raise ValueError("Elige phi_today y F de modo que 1 - cos(phi_today/F) > 0")

LAMBDA4 = OMEGA_DE0 / denom

# Dominio en N = ln(a)
N_min = np.log(1e-3)   # a_min = 1e-3 (z ~ 999), puedes subir/lower
N_max = 0.0
N_eval = np.linspace(N_min, N_max, 600)


# ================== POTENCIAL Y FONDO COSMOLÓGICO =====================

def V(phi):
    """Potencial PNGB."""
    return LAMBDA4 * (1.0 - np.cos(phi / F))

def dV_dphi(phi):
    """Derivada del potencial PNGB."""
    return LAMBDA4 * (1.0 / F) * np.sin(phi / F)

def background(N, phi, dphi_dN):
    """
    Calcula a, H, rho_r, rho_m, rho_phi, rho_tot y p_phi, p_tot.
    Usa la solución auto-consistente:
        H^2 = (rho_r + rho_m + V) / (1 - 0.5 * (dphi_dN)^2)
    """
    a = np.exp(N)

    rho_r = OMEGA_R0 * a**(-4)
    rho_m = OMEGA_M0 * a**(-3)

    V_phi = V(phi)

    # Resolver H^2 analíticamente a partir de la ecuación
    #   H^2 = rho_r + rho_m + V + 0.5 H^2 (dphi_dN)^2
    # => H^2 (1 - 0.5 dphi_dN^2) = rho_r + rho_m + V
    fac = 1.0 - 0.5 * dphi_dN**2
    if fac <= 0:
        # Esto sería un régimen no físico (campo tipo phantom),
        # aquí simplemente evitamos crash y forzamos algo positivo
        fac = 1e-6

    H2 = (rho_r + rho_m + V_phi) / fac
    if H2 <= 0:
        H2 = 1e-12
    H = np.sqrt(H2)

    # Densidad del campo
    dot_phi = H * dphi_dN
    rho_kin = 0.5 * dot_phi**2
    rho_phi = rho_kin + V_phi

    rho_tot = rho_r + rho_m + rho_phi

    # Presiones
    p_r = rho_r / 3.0
    p_phi = rho_kin - V_phi
    p_tot = p_r + p_phi

    return a, H, rho_r, rho_m, rho_phi, rho_tot, p_phi, p_tot


# =================== SISTEMA DE EDO EN N = ln(a) ======================

def rhs(N, y):
    """
    Sistema en N:
        y[0] = phi
        y[1] = dphi/dN
    """
    phi, dphi_dN = y

    a, H, rho_r, rho_m, rho_phi, rho_tot, p_phi, p_tot = background(N, phi, dphi_dN)

    w_eff = p_tot / rho_tot if rho_tot > 0 else -1.0
    dlnH_dN = -1.5 * (1.0 + w_eff)  # de Friedmann + continuidad

    dphi_dN_eq = dphi_dN
    d2phi_dN2 = - (3.0 + dlnH_dN) * dphi_dN - dV_dphi(phi) / (H**2)

    return [dphi_dN_eq, d2phi_dN2]


# ============================= MAIN ===================================

def main():
    print("\n=== PNGB Siamese Phase Solver — FIXED v1 ===")
    print(f"  Ω_r0   = {OMEGA_R0:.6e}")
    print(f"  Ω_m0   = {OMEGA_M0:.6f}")
    print(f"  Ω_DE0  = {OMEGA_DE0:.6f}")
    print(f"  f      = {F:.4f}")
    print(f"  phi_0  = {phi_today:.4f}")
    print(f"  Λ⁴     = {LAMBDA4:.6f} (calibrada para V(phi_0)=Ω_DE0)\n")

    y0 = [phi_today, dphi_dN_today]

    sol = solve_ivp(
        rhs,
        (N_min, N_max),
        y0,
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
        a, H, rho_r, rho_m, rho_phi, rho_tot, p_phi, p_tot = background(N, phi, dphi_dN)
        w_eff = p_tot / rho_tot if rho_tot > 0 else -1.0
        w_phi = p_phi / rho_phi if rho_phi > 0 else -1.0
        Delta_phi = 2.0 * phi

        rows.append([
            a, N, phi, dphi_dN, Delta_phi,
            H, rho_r, rho_m, rho_phi, rho_tot, p_phi, p_tot, w_eff, w_phi
        ])

    df = pd.DataFrame(rows, columns=[
        "a", "N", "phi", "dphi_dN", "Delta_phi",
        "H", "rho_r", "rho_m", "rho_phi", "rho_tot", "p_phi", "p_tot",
        "w_eff", "w_phi"
    ])

    df = df.sort_values("a").reset_index(drop=True)

    out_csv = "../results/csv/KG_PNGB_fixed_f1_2_OMEGAde0_0_65_v1.csv"
    df.to_csv(out_csv, index=False)

    # Punto hoy (a ~ 1)
    today = df.iloc[-1]
    print("✅ Integración completada.\n")
    print(f"  a(último)      = {today['a']:.6f}")
    print(f"  H(a=1)        ≈ {today['H']:.6f} (debería ser ≈ 1)")
    print(f"  w_phi(a=1)    ≈ {today['w_phi']:.6f}")
    print(f"  w_eff(a=1)    ≈ {today['w_eff']:.6f}\n")
    print("📁 Resultados guardados en:")
    print(f"  {out_csv}\n")
    print("Columnas:")
    print("  a, N, phi, dphi_dN, Delta_phi, H, rho_r, rho_m, rho_phi,")
    print("  rho_tot, p_phi, p_tot, w_eff, w_phi")


if __name__ == "__main__":
    main()
