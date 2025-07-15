import math
from math import sqrt
from typing import Optional

import numpy as np

from .data import THICKNESS_TABLE, BV_VALUE_DF, E_VALUE_DF


def factored_compressive_resistance(n: int, d: float, L: float) -> float:
    phi = 0.8
    f_c = 11.5
    K_D = 1.15
    K_H = 1.0
    K_Sc = 1.0
    K_T = 1.0
    K_SE = 1.0
    b = 38
    A = n * b * d
    E_05 = 6500
    C_cd = L / d
    K_zcd = 6.3 * (d * L) ** -0.13
    F_c = f_c * (K_D * K_H * K_Sc * K_T)
    K_cd = (1 + (F_c * K_zcd * C_cd ** 3) / (35 * E_05 * K_SE * K_T)) ** -1
    P_r = phi * F_c * A * K_zcd * K_cd
    return (0.6 * P_r) / 1000


def unit_lateral_strength_resistance(d_f, t_1, t_2, J_x, G, G_f3):
    f_y = 50 * (16 - d_f)
    f_1 = 51 * (1 - 0.1 * d_f)
    f_2 = 50 * G * (1 - 0.01 * d_f) * J_x
    f_3 = 110 * G_f3 ** 1.8 * (1 - 0.01 * d_f) * J_x
    a = f_1 * d_f * t_1
    b = f_2 * d_f * t_2
    d = f_1 * d_f ** 2 * (sqrt((f_y * f_3) / (6 * (f_1 + f_3) * f_1)) + (t_1 / (5 * d_f)))
    e = f_1 * d_f ** 2 * (sqrt((f_y * f_3) / (6 * (f_1 + f_3) * f_1)) + (t_2 / (5 * d_f)))
    f = f_1 * d_f ** 2 * (1 / 5) * ((t_1 / d_f) + ((f_2 / f_1) * (t_2 / d_f)))
    g = f_1 * d_f ** 2 * sqrt((2 * f_3 * f_y) / (3 * (f_1 + f_3) * f_1))
    return min(a, b, d, e, f, g)


def fastener_spacing_factor(s):
    return 1.0 if s >= 150 else 1 - ((150 - s) / 150) ** 4.2


def sheathing_to_framing_shear(phi, N_u, s, J_D, n_s, J_us, J_s, J_hd):
    return phi * (N_u / s) * J_D * n_s * J_us * J_s * J_hd


def get_data_by_thickness(thickness: float):
    for data in THICKNESS_TABLE:
        if data.nominal_thickness == thickness:
            return data
    return None


def panel_buckling_strength(a, b, t, B_v, B_a0, B_a90):
    alpha = (a / b) * (B_a90 / B_a0) ** (1 / 4)
    eta = (2 * B_v) / sqrt(B_a0 * B_a90)
    k_pb = 1.7 * (eta + 1) * math.exp((-alpha / (0.05 * eta + 0.75))) + (0.5 * eta + 0.8)
    v_pb = k_pb * ((math.pi ** 2 * t ** 2) / (3000 * b)) * (B_a0 * B_a90 ** 3) ** (1 / 4)
    return alpha, eta, k_pb, v_pb


def panel_buckling_shear(phi, K_D, K_s, K_T, v_pb):
    return phi * v_pb * K_D * K_s * K_T


def get_bv(panel_type: str, thickness: float) -> Optional[int]:
    result = BV_VALUE_DF[(BV_VALUE_DF["Panel Type"] == panel_type) & (BV_VALUE_DF["Thickness"] == thickness)]
    if not result.empty:
        return int(result.iloc[0]["BV"])
    return None


def get_elasticity(species: str, grade: str):
    result = E_VALUE_DF[(E_VALUE_DF["Species"] == species) & (E_VALUE_DF["Grade"] == grade)]
    if not result.empty:
        return result.iloc[0]["Epar"], result.iloc[0]["Eperp"]
    return None, None


def deflection_due_to_bending(V, H, EI_tr, M):
    return ((V * H ** 3) / (3 * EI_tr)) + ((M * H ** 2) / (2 * EI_tr))


def deflection_due_to_panel_shear(V, H, L, B_v):
    return (V * H) / (L * B_v)


def deflection_due_to_wall_anchorage(d_a, H, L_s):
    return (H / L_s) * d_a


def deflection_due_to_rotation_at_bottom(H_i, theta_all, alpha_all):
    return H_i * (theta_all + alpha_all)


def total_deflection(D_bending, D_panel_shear, D_nail_slip, D_wall_anchorage, D_rotation_bottom):
    return D_bending + D_panel_shear + D_nail_slip + D_wall_anchorage + D_rotation_bottom
