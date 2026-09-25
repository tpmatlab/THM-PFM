from dolfin import *

################################## Constants ################################## 
epsilon_emi = Constant(0.56)  # Emissivity of the material
sigma_SB = Constant(5.67e-8)  # Stefan-Boltzmann constant
biot = Constant(1.0)  # Biot number
alpha_T = 0.0  # Thermal expansion coefficient
nu = 0.25  # Poisson's Ratio
l_0 = Constant(0.0041)  # Characteristic length for phase-field model

model = "plane_strain"  # Model type for elasticity calculations
E_Constant = True  # Flag to indicate if Young's modulus is constant
Gc_Constant = True  # Flag to indicate if fracture energy is constant

######################## Elastic properties of concrete ########################
if E_Constant:
    def E(T_K):
        E_0 = 14e9
        E_val = E_0
        return E_val


else:
    def E(T_K):
        E_0 = 14e9
        T = T_K - 273.15
        E_20_300 = (1 - 0.0015 * T) * E_0
        E_300_700 = (0.87 - 0.0084 * T) * E_0
        E_700 = 0.28 * E_0
        E_val = conditional(le(T, 300), E_20_300,
                        conditional(le(T, 700), E_300_700,
                                    E_700))
        return E_val


if model == "plane_stress":
    print(f'Model: Plane Stress')


elif model == "plane_strain":
    print(f'Model: Plane Strain')


def mu(T):
    mu_val = E(T) / (2 * (1 + nu))
    return mu_val


def lmbda(T):
    lmbda_0 = E(T) * nu / ((1 + nu) * (1 - 2 * nu))
    if model == "plane_stress":
        lmbda_val = (2 * mu(T) * lmbda_0 / (lmbda_0 + 2 * mu(T)))
    elif model == "plane_strain":
        lmbda_val = lmbda_0
    return lmbda_val


if Gc_Constant:
    def Gc(T_K):
        Gc_0 = 95  # in MPa m = N/m
        Gc_val = Gc_0
        return Gc_val


else:
    def Gc(T_K):
        Gc_0 = 95  # in MPa m = N/m
        T = T_K - 273.15
        gamma_a = 15.4
        T_0 = 298.15
        Gc_val = Gc_0 * exp(gamma_a * (1 / T - 1 / T_0))
        return Gc_val


def K(T):
    K_val = (3 * lmbda(T) + 2 * mu(T)) / 3  # Bulk Modulus
    return K_val


##################### Thermohygro properties of concrete #######################
def P_s(T):
    C_1 = -5800.2206
    C_2 = 1.3914993
    C_3 = -4.8640239e-2
    C_4 = 4.1764768e-5
    C_5 = -1.4452093e-8
    C_6 = 6.5459673
    Pvps_vals = exp(C_1 / T + C_2 + C_3 * T + C_4 * T**2 + C_5 * T**3 +
                    C_6 * ln(T))
    return Pvps_vals

def lambda_c(Pv, T):
    lambda_c_vals = Constant(1.67)
    return lambda_c_vals


# Relative humidity
def h(P_v, T):
    return P_v / P_s(T)


# Properties of concrete
# Dehydration water
def w_d(T):
    x = T - 273.15
    A1 = 22.838987236074114
    A2 = 0.1392389207591992
    x0 = 271.38555604747785
    dx = 23.059326752940066
    w_d_vals = A1 + (A2 - A1) / (1 + exp((x - x0) / dx))
    return w_d_vals


def dw_ddt(T, T_n, dt):
    delta = 1e-8
    dT = delta * T_n
    dw_ddT = ((w_d(T_n + dT / 2) - w_d(T_n - dT / 2)) / (dT))
    dTdt = (T - T_n) / dt
    dw_ddt_vals = dw_ddT * dTdt
    return dw_ddt_vals


def rho_s(T):
    rho_vals = Constant(2200)
    return rho_vals


# Sorption isotherm
w_c = Constant(300.0)  # Cement content
w_0 = Constant(100)    # Water content
a_0 = Constant(1e-14)  # Initial hydraulic conductivity

def m(T):
    return 1.04 - ((T - 263.15)**2) / ((T - 263.15)**2 +
                                       22.34 * (298.15 - 263.15)**2)


def w_1(P_v, T):
    return w_c * ((w_0 / w_c) * (h(P_v, T)))**(1 / m(T))


def w_2(P_v, T):
    return w_c * (0.037 * (h(P_v, T) - 1.04) +
                  0.3335 * (1.0 - ((T - 273.15)**2.0) / (3.6e5)))


def w(P_v, T):
    w_096 = w_1(0.96 * P_s(T), T)
    w_104 = w_2(1.04 * P_s(T), T)
    w_int = w_096 + ((w_104 - w_096) * (h(P_v, T) - 0.96) / 0.08)
    return conditional(le(T, 647.15),
                   conditional(le(h(P_v, T), 0.96), w_1(P_v, T),
                               conditional(le(h(P_v, T), 1.04), w_int,
                                           w_2(P_v, T))), 0.0)

def dwdt(P_v, T, P_v_n, T_n, dt):
    delta = 1e-4
    dP = delta * P_v_n
    dT = delta * T_n
    dwdP = ((w(P_v_n + dP / 2, T_n) - w(P_v_n - dP / 2, T_n)) / (dP))
    dwdT = ((w(P_v_n, T_n + dT / 2) - w(P_v_n, T_n - dT / 2)) / (dT))
    dPdt = (P_v - P_v_n) / dt
    dTdt = (T - T_n) / dt
    return dwdP * dPdt + dwdT * dTdt


# Latent heat of dehydration
def DeltaH_d(T):
    DeltaH_d_vals = Constant(0.0)
    return DeltaH_d_vals


# Latent heat of evaporation of water
def DeltaH_ev(T):
    DeltaH_ev_vals = conditional(le(T, 647.3),
                                 3.5e5 * ((647.3 - T)**(1 / 3)), 0.0)
    return DeltaH_ev_vals


# Specific heat of solid refractory
def C_p(T):
    C_p_vals = Constant(1100)
    return C_p_vals


# Specific heat of water
def C_p_l(T):
    C_p_l_vals = Constant(4100.0)
    return C_p_l_vals


# Permeability
def a_t(T):
    return (T - 273.15) * 0.95 / 70. - 0.28928571


def f1(T, h):
    return conditional(lt(h, 1), (a_t(T) + (1. - a_t(T)) / (1. + (4. * (1. - h))**4.)), 1)


def f2(T):
    return exp(2700 * (1. / (273.15 + 25.) - 1. / (T)))


def f3(T):
    return exp(((T - 273.15) - 95.) /
               (0.881 + 0.214 * ((T - 273.15) - 95.)))


def a(P_v, T):
    return conditional(le(T, 368.15), a_0 * f1(T, h(P_v, T)) * f2(T),
                       a_0 * 5.6 * f3(T))