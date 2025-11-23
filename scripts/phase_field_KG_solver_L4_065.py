import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# ==============================================================================
# 1. PARÁMETROS COSMOLÓGICOS Y DEL POTENCIAL (unidades normalizadas)
# ==============================================================================

# Radiación (Planck-like)
OMEGA_R0 = 9.0e-5

# Potencial PNGB: V(phi) = LAMBDA4 * [1 - cos(phi / F)]
F = 0.8          # escala de simetría
LAMBDA4 = 0.65   # <<--- NUEVO VALOR CALIBRADO

# Materia actual emergente (no la tocamos a mano, solo derivamos)
OMEGA_M0 = 1.0 - OMEGA_R0 - LAMBDA4

# Condiciones iniciales a(N=0) ≡ a=1
PHI_0 = 0.2
UPHI_0 = 0.0
Y0 = [PHI_0, UPHI_0]

# Rango en N = ln a (de hoy hacia atrás)
A_MIN = 0.01
N_START = 0.0
N_END = np.log(A_MIN)
N_POINTS = 500
N_EVAL = np.linspace(N_START, N_END, N_POINTS)


# ==============================================================================
# 2. POTENCIAL Y DENSIDADES
# ==============================================================================

def potential(phi: float) -> float:
    return LAMBDA4 * (1.0 - np.cos(phi / F))


def potential_prime(phi: float) -> float:
    return LAMBDA4 * (1.0 / F) * np.sin(phi / F)


def densities_and_H2(N: float, phi: float, uphi: float):
    a = np.exp(N)

    rho_r = OMEGA_R0 * a**(-4.0)
    rho_m = OMEGA_M0 * a**(-3.0)
    V = potential(phi)

    denom = 1.0 - 0.5 * uphi**2
    if denom <= 1e-8:
        denom = 1e-8

    H2 = (rho_r + rho_m + V) / denom

    rho_phi = 0.5 * H2 * uphi**2 + V
    p_phi = 0.5 * H2 * uphi**2 - V

    rho_tot = rho_r + rho_m + rho_phi
    p_tot = rho_r / 3.0 + p_phi

    w_eff = p_tot / rho_tot

    return {
        "a": a,
        "rho_r": rho_r,
        "rho_m": rho_m,
        "rho_phi": rho_phi,
        "p_phi": p_phi,
        "rho_tot": rho_tot,
        "p_tot": p_tot,
        "w_eff": w_eff,
        "H2": H2,
    }


# ==============================================================================
# 3. ECUACIONES DIFERENCIALES EN N = ln a
# ==============================================================================

def deriv(N: float, y):
    phi, uphi = y

    cosmo = densities_and_H2(N, phi, uphi)
    w_eff = cosmo["w_eff"]
    H2 = cosmo["H2"]

    dphi_dN = uphi
    duphi_dN = -1.5 * (1.0 - w_eff) * uphi - potential_prime(phi) / H2

    return [dphi_dN, duphi_dN]


# ==============================================================================
# 4. INTEGRACIÓN NUMÉRICA
# ==============================================================================

if __name__ == "__main__":
    print("🔄 Integrando ecuación de fase PNGB en fondo cosmológico (N = ln a)...")
    print(f"   Parámetros: F = {F}, LAMBDA4 = {LAMBDA4}, OMEGA_R0 = {OMEGA_R0}")
    print(f"   OMEGA_M0 (emergente) = {OMEGA_M0:.6f}")

    sol = solve_ivp(
        fun=deriv,
        t_span=(N_START, N_END),
        y0=Y0,
        t_eval=N_EVAL,
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
    )

    if not sol.success:
        print("❌ Error en la integración:", sol.message)
    else:
        print("✅ Integración completada correctamente.")

        N_arr = sol.t
        phi_arr = sol.y[0]
        uphi_arr = sol.y[1]

        rows = []
        for N, phi, uphi in zip(N_arr, phi_arr, uphi_arr):
            cosmo = densities_and_H2(N, phi, uphi)
            a = cosmo["a"]
            H2 = cosmo["H2"]
            H = np.sqrt(H2)

            rho_r = cosmo["rho_r"]
            rho_m = cosmo["rho_m"]
            rho_phi = cosmo["rho_phi"]
            p_phi = cosmo["p_phi"]
            rho_tot = cosmo["rho_tot"]
            p_tot = cosmo["p_tot"]
            w_eff = cosmo["w_eff"]

            rho_DE = rho_phi
            Omega_DE = rho_DE / H2
            w_phi = p_phi / rho_phi if rho_phi > 0 else np.nan
            Delta_phi = 2.0 * phi

            rows.append([
                a, N, phi, uphi, Delta_phi,
                H, rho_r, rho_m, rho_DE, Omega_DE,
                rho_tot, p_tot, w_eff, w_phi
            ])

        df = pd.DataFrame(
            rows,
            columns=[
                "a", "N", "phi", "dphi_dN", "Delta_phi",
                "H", "rho_r", "rho_m", "rho_DE", "Omega_DE",
                "rho_tot", "p_tot", "w_eff", "w_phi"
            ],
        )

        df = df.sort_values("a").reset_index(drop=True)

        # Nombre del archivo NUEVO (nota: sin .csv por el replace)
        filename = f"KG_PNGB_f{F}_L4_{LAMBDA4}_phi0_{PHI_0}.csv".replace(".", "_")
        out_path = f"../results/csv/{filename}"
        df.to_csv(out_path, index=False)

        print("")
        print("📁 Resultados guardados en:")
        print(f"   {out_path}")
        print("")
        print("Columnas disponibles en el CSV:")
        print("   a, N, phi, dphi_dN, Delta_phi, H, rho_r, rho_m, rho_DE, Omega_DE,")
        print("   rho_tot, p_tot, w_eff, w_phi")
