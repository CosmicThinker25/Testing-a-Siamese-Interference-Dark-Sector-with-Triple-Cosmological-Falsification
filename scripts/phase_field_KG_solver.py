import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# ==============================================================================
# 1. PARÁMETROS COSMOLÓGICOS Y DEL POTENCIAL (unidades normalizadas)
#    - No imponemos explícitamente H0 = 1, dejamos que H(0) salga del sistema.
#    - Trabajamos en unidades en las que H^2 ∝ rho_total.
# ==============================================================================

# Radiación (valor típico Planck 2018 en unidades normalizadas)
OMEGA_R0 = 9.0e-5

# Amplitud del potencial tipo PNGB: V(phi) = LAMBDA4 * [1 - cos(phi / F)]
# Aquí LAMBDA4 juega el papel de "escala" efectiva de energía oscura.
LAMBDA4 = 0.70

# Materia actual ajustada para un universo aproximadamente plano
# (no exigimos que V(phi0) = LAMBDA4 exactamente; dejamos que H0 salga del sistema)
OMEGA_M0 = 1.0 - OMEGA_R0 - LAMBDA4

# Parámetros del potencial PNGB
F = 0.8  # escala de simetría (en unidades reducidas)

# Condiciones iniciales a(N=0) ≡ a=1
PHI_0 = 0.2       # valor del campo hoy
UPHI_0 = 0.0      # dphi/dN hoy (campo casi congelado)
Y0 = [PHI_0, UPHI_0]

# Rango de integración en N = ln a
# Integramos hacia atrás hasta a_min ~ 0.01 (z ~ 100)
A_MIN = 0.01
N_START = 0.0                 # N = ln(a), hoy
N_END = np.log(A_MIN)         # N negativo
N_POINTS = 500
N_EVAL = np.linspace(N_START, N_END, N_POINTS)


# ==============================================================================
# 2. POTENCIAL Y DENSIDADES
# ==============================================================================

def potential(phi: float) -> float:
    """
    Potencial PNGB:
        V(phi) = LAMBDA4 * [1 - cos(phi / F)]
    """
    return LAMBDA4 * (1.0 - np.cos(phi / F))


def potential_prime(phi: float) -> float:
    """
    Derivada del potencial:
        V'(phi) = LAMBDA4 * (1/F) * sin(phi / F)
    """
    return LAMBDA4 * (1.0 / F) * np.sin(phi / F)


def densities_and_H2(N: float, phi: float, uphi: float):
    """
    Calcula densidades y H^2 de forma autoconsistente a partir de:
      - N = ln a
      - phi(N)
      - uphi(N) = dphi/dN

    Usamos:
      a = exp(N)
      rho_r = OMEGA_R0 * a^-4
      rho_m = OMEGA_M0 * a^-3
      H^2 = (rho_r + rho_m + V) / (1 - 0.5 * uphi^2)
      rho_phi = 0.5 * H^2 * uphi^2 + V
      p_phi   = 0.5 * H^2 * uphi^2 - V
    """
    a = np.exp(N)

    # Densidades estándar
    rho_r = OMEGA_R0 * a**(-4.0)
    rho_m = OMEGA_M0 * a**(-3.0)

    # Potencial del campo de fase
    V = potential(phi)

    # Resolver H^2 del constraint de Friedmann con campo escalar:
    # H^2 = rho_r + rho_m + (0.5 H^2 uphi^2 + V)
    # => H^2 (1 - 0.5 uphi^2) = rho_r + rho_m + V
    denom = 1.0 - 0.5 * uphi**2

    # Protección mínima numérica por si uphi^2 → 2
    if denom <= 1e-8:
        denom = 1e-8

    H2 = (rho_r + rho_m + V) / denom

    # Energía y presión del campo
    rho_phi = 0.5 * H2 * uphi**2 + V
    p_phi = 0.5 * H2 * uphi**2 - V

    # Totales
    rho_tot = rho_r + rho_m + rho_phi
    p_tot = rho_r / 3.0 + p_phi

    # Parámetros efectivos
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
    """
    Sistema de EDOs en N = ln(a):

      y[0] = phi
      y[1] = uphi = dphi/dN

    Ecuación de KG en N:
      phi'' + (3 + H'/H) phi' + V'(phi)/H^2 = 0

    Usamos:
      H'/H = - (3/2) (1 + w_eff)

    ⇒ uphi' = - [3 + H'/H] uphi - V'/H^2
            = - 1.5 (1 - w_eff) uphi - V'/H^2
    """
    phi, uphi = y

    # Calcular densidades, H^2 y w_eff para este N
    cosmo = densities_and_H2(N, phi, uphi)
    w_eff = cosmo["w_eff"]
    H2 = cosmo["H2"]

    # Derivadas
    dphi_dN = uphi
    duphi_dN = -1.5 * (1.0 - w_eff) * uphi - potential_prime(phi) / H2

    return [dphi_dN, duphi_dN]


# ==============================================================================
# 4. INTEGRACIÓN NUMÉRICA
# ==============================================================================

if __name__ == "__main__":
    print("🔄 Integrando ecuación de fase PNGB en fondo cosmológico (N = ln a)...")

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

        # Extraer soluciones
        N_arr = sol.t
        phi_arr = sol.y[0]
        uphi_arr = sol.y[1]

        # Calcular magnitudes derivadas para cada punto
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

            # Energía oscura del campo (lo identificamos con rho_phi)
            rho_DE = rho_phi
            Omega_DE = rho_DE / H2

            # Ecuación de estado del DE solo (w_phi)
            w_phi = p_phi / rho_phi if rho_phi > 0 else np.nan

            # Desfase interferencial
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

        # Ordenar por a ascendente (del universo temprano al tardío)
        df = df.sort_values("a").reset_index(drop=True)

        # Guardar CSV en la carpeta de resultados
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
        print("")
        print("🔥 Próximo paso: usar este CSV en generate_figures_v2.py para producir:")
        print("   - rho_eff(a), rho_DM(a), rho_DE(a)")
        print("   - H(a) y H(z) vs ΛCDM")
        print("   - w_phi(a) y w_eff(a)")
        print("   - Δphi(a) y entropía siamés.")
