from dolfin import *
from material_properties import *
from datetime import datetime
import csv
import os
import sys
import numpy as np
from shutil import copy
from sys import argv
from functools import partial
from utils import *


###############################################################################
################################ 1. Case Setup ################################
###############################################################################

# Update solver parameters
parameters, ffc_options = update_params(parameters, has_linear_algebra_backend)

# Name of the case
case = f'case_1_column_substepping'
# Create the directory for output files
dir_case = setup_case_directory(case)

# Set FEniCS log level and redirect log to log file
set_log_level(50)
sys.stdout = Logger(dir_case + '/log_' + case + '.log')

# Copy the code for backup
copy(argv[0], dir_case + '/' + case + '_code_backup.py')

# Print the name of the case
print(f'    | Running case: ' + ' '.join([word.capitalize() for word
                                          in case.replace('_', ' ').split()]))


###############################################################################
########################### 2. Time Discretization ############################
###############################################################################

t = 0
dt = 5
T_total = 280
print(f'    |- Time discretization:')
print(f'    |-- Total time = {round(T_total / 3600)} hrs')
print(f'    |-- dt = {dt} s')


###############################################################################
###################### 3. Spatial Discretization (Mesh) #######################
###############################################################################

lx = ly = 0.20              
Nx = Ny = 200

print("    |- Geometry:")
print(f'    |-- lx = {lx}m | ly = {ly}m | nx = {Nx} | ny = {Ny}')

x_0_y_0 = Point(0, 0)
x_1_y_1 = Point(lx, ly)
mesh = RectangleMesh.create([x_0_y_0, x_1_y_1], [Nx, Ny],
                            CellType.Type.quadrilateral)

print("    |- Mesh:")
print("    |-- Number of vertices = " + str(mesh.num_vertices()))
print("    |-- Number of cells = " + str(mesh.num_cells()))
print("    |-- Cell size hmax, hmin = %.3g %.3g" % (mesh.hmax(), mesh.hmin()))

###############################################################################
############################ 4. Boundaries Marking ############################
###############################################################################


boundaries = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)

class left(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0]) < DOLFIN_EPS and on_boundary


class right(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - lx) < DOLFIN_EPS and on_boundary


class top(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[1] - ly) < DOLFIN_EPS and on_boundary


class bottom(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[1]) < DOLFIN_EPS and on_boundary


left = left()
right = right()
top = top()
bottom = bottom()

left.mark(boundaries, 1)
right.mark(boundaries, 2)
top.mark(boundaries, 3)
bottom.mark(boundaries, 4)

File(dir_case + '/boundaries_marking_' + case + '.pvd') << boundaries

ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
ds_heated = (ds(2) + ds(3))


###############################################################################
##################### 5. Thermohygral Problem Definition ######################
###############################################################################

# Mixed finite element and space function definition for the thermal problem
k_P = 1
k_T = 1
Pk_P = FiniteElement('P', mesh.ufl_cell(), k_P)
Pk_T = FiniteElement('P', mesh.ufl_cell(), k_T)
element = MixedElement([Pk_P, Pk_T])
V = FunctionSpace(mesh, element)
q_P, q_T = TestFunctions(V)
r = Function(V)
P_v, T = split(r)
r_n = Function(V)
P_v_n, T_n = split(r_n)
r_n_i = Function(V)
P_v_n_i, T_n_i = split(r_n_i)

# Initial condition (IC)
P_0 = Constant(1020.0)
P_v_n = interpolate(P_0, V.sub(0).collapse())
P_v_n_i = interpolate(P_0, V.sub(0).collapse())

T_0 = Constant(298.15)
T_n = interpolate(T_0, V.sub(1).collapse())
T_n_i = interpolate(T_0, V.sub(1).collapse())

print(f'    |- Initial conditions:')
print(f'    |-- T_0 = {round(T_0  - 273.15, 2)}°C')
print(f'    |-- Pv_0 = {round(float(P_0), 2)} Pa')
print(f'    |-- RH_0 = {round(float(P_0)/float(P_s(T_0)) * 100, 2)}%')
print(f'    |-- S_l_0 = {round(float(w(P_0, T_0) / (0.1 * 1000)), 2)}')


# Thermal problem boundary conditions (BC)
B_t = Constant(25.0)        # Thermal transfer coefficient
T_inf = Constant(298.15)    # Surrounding environment temperature
B_w = Constant(0.025)       # Mass transfer coefficient 
P_v_inf = Constant(1020.0)  # Surrounding environment pressure

# Heat-up curve
def RABT(t):
    T_final = 1473.15
    if t < 300:
        t_0 = 0
        T_0 = 298.15
        return (t - t_0) * (T_final - T_0) / (300) + T_0

    else:
        return T_final

T_flame = Expression('T', degree=1, T=RABT(0))

# Thermohygro problem Dirichlet boundary conditions (BC)
bcs_TH = []

# Variational formulation of the thermal problem in residual form
ResP = dwdt(P_v, T, P_v_n, T_n, dt) * q_P * dx
ResP += (a(P_v_n, T_n)) * inner(nabla_grad(P_v), nabla_grad(q_P)) * dx
ResP += - dw_ddt(T, T_n, dt) * q_P * dx
ResP += B_w * (P_v - P_v_inf) * q_P * (ds(2) + ds(3))


# Energy balance equation
ResT = rho_s(T_n) * C_p(T_n) * ((T - T_n) / dt) * q_T * dx
ResT += lambda_c(P_v_n, T) * inner(nabla_grad(T), nabla_grad(q_T)) * dx
ResT += + (DeltaH_d(T)) * dw_ddt(T, T_n, dt) * q_T * dx
ResT += - DeltaH_ev(T_n) * dwdt(P_v, T, P_v_n, T_n, dt) * q_T * dx
ResT += C_p_l(T_n) * (a(P_v_n, T_n)) * \
        inner(nabla_grad(P_v_n), nabla_grad(T_n)) * q_T * dx

# Robin boundary conditions for energy balance equation
ResT += (B_t * (T - T_flame) + epsilon_emi * sigma_SB * (T_n**3 * T - T_flame**4)) * q_T * (ds(2) + ds(3))

# Total residual
Res_TH = ResT + ResP

# Jacobian
Jac_TH = derivative(Res_TH, r)

# Nonlinear th_problem and th_solver parameters
problem_TH = NonlinearVariationalProblem(Res_TH, r, bcs_TH, Jac_TH, ffc_options)
solver_TH = NonlinearVariationalSolver(problem_TH)
prm_TH = solver_TH.parameters
prm_TH["newton_solver"]["absolute_tolerance"] = 1E-7
prm_TH["newton_solver"]["relative_tolerance"] = 1E-13
prm_TH["newton_solver"]["maximum_iterations"] = 15
prm_TH['newton_solver']['error_on_nonconvergence'] = True


###############################################################################
################### 6. Linear Elastic - Phase-field Problem ###################
###############################################################################

# Vector function space for the displacement problem
k_u = 1
W = VectorFunctionSpace(mesh, 'CG', k_u)
u = Function(W)
u_n = Function(W)
v = TestFunction(W)

# Scalar function space for the phase-field problem
k_phi = 1
V_phi = FunctionSpace(mesh, 'CG', k_phi)
phi_hat = TrialFunction(V_phi)
phi = Function(V_phi)
phi_n = Function(V_phi)
q_phi = TestFunction(V_phi)

# Discontinous Galerkin function space for the History variable
k_H = 0
V_H = FunctionSpace(mesh, 'DG', k_H)
Hold = Function(V_H)

# Tensor Function Space to project the principal stresses
TS = TensorFunctionSpace(mesh, "CG", 2) 


def epsilon(u):
    '''
    Total strain tensor assuming small deformations.
    
    Parameters
    ----------
    u : FEniCS vector function
        Displacement field.

    Returns
    -------
    epsilon_val : FEniCS function
        Strain tensor.
    '''

    epsilon_val = 0.5*(grad(u) + grad(u).T)
    return epsilon_val


def epsilon_thermal(u, T):
    '''
    Thermal strain tensor.
    
    Parameters
    ----------
    u : FEniCS vector function
        Displacement field.

    T : FEniCS scalar function
        Temperature field.

    Returns
    -------
    epsilon_T_val : FEniCS function
        Thermal strain tensor.
    '''
    epsilon_T_val = alpha_T * (T - T_0) * Identity(len(u))
    return epsilon_T_val


def epsilon_hygral(u, P_v, T):
    '''
    Hygral strain tensor.
    
    Parameters
    ----------
    u : FEniCS vector function
        Displacement field.

    P_v : FEniCS scalar function
        Pressure field.

    T : FEniCS scalar function
        Temperature field.

    Returns
    -------
    epsilon_P_val : FEniCS function
        Hygral strain tensor.
    '''
    epsilon_P_val = biot / K(T) * (P_v) * Identity(len(u))
    return epsilon_P_val


def epsilon_elastic(u, P_v, T):
    '''
    Elastic strain tensor.
    
    Parameters
    ----------
    u : FEniCS vector function
        Displacement field.

    P_v : FEniCS scalar function
        Pressure field.

    T : FEniCS scalar function
        Temperature field.

    Returns
    -------
    epsilon_e_val : FEniCS function
        Elastic strain tensor.
    '''
    epsilon_e_val = epsilon(u) - epsilon_thermal(u, T) - epsilon_hygral(u, P_v, T)
    return epsilon_e_val
    

def psi_D(eps_e, P_v, T):
    '''
    Degradable elastic strain energy density. The only component that 
    contributes to the energy release rate.
    
    Parameters
    ----------
    eps_e: FEniCS function
        Elastic strain.

    P_v : FEniCS scalar function
        Pressure field.

    T : FEniCS scalar function
        Temperature field.

    Returns
    -------
    psi_D_val : FEniCS function
        The degradable elastic strain energy density.


    Notes
    -----
    Using the spectral split.
    Following the nomenclature by Vicentini et al. 
    HAL Id: hal-04231075
    https://hal.sorbonne-universite.fr/hal-04231075

    The spectral decomposition follows the approach by Xue et al.
    https://doi.org/10.1016/j.cma.2021.114046
    https://github.com/tianjuxue/crack/blob/main/src/constitutive.py#L79
    '''
    sqrt_delta = conditional(gt(tr(eps_e)**2 - 4 * det(eps_e), 0),
                             sqrt(tr(eps_e)**2 - 4 * det(eps_e)),
                             0)
    eigen_value_1 = (tr(eps_e) + sqrt_delta) / 2
    eigen_value_2 = (tr(eps_e) - sqrt_delta) / 2
    psi_D_val = (lmbda(T) / 2 * ppo(tr(eps_e))**2 +
                 mu(T) * (ppo(eigen_value_1)**2 + ppo(eigen_value_2)**2))
    return psi_D_val


def psi_R(eps_e, P_v, T):
    '''
    Residual elastic strain energy density.

    Parameters
    ----------
    eps_e: FEniCS function
        Elastic strain.

    P_v : FEniCS scalar function
        Pressure field.

    T : FEniCS scalar function
        Temperature field.

    Returns
    -------
    psi_R_val : FEniCS function
        The residual elastic strain energy density.

    Notes
    -----
    Using the spectral split.
    Following the nomenclature by Vicentini et al. 
    HAL Id: hal-04231075
    https://hal.sorbonne-universite.fr/hal-04231075

    The spectral decomposition follows the approach by Xue et al.
    https://doi.org/10.1016/j.cma.2021.114046
    https://github.com/tianjuxue/crack/blob/main/src/constitutive.py#L79
    '''
    sqrt_delta = conditional(gt(tr(eps_e)**2 - 4 * det(eps_e), 0),
                             sqrt(tr(eps_e)**2 - 4 * det(eps_e)),
                             0)
    eigen_value_1 = (tr(eps_e) + sqrt_delta) / 2
    eigen_value_2 = (tr(eps_e) - sqrt_delta) / 2
    psi_R_val = (lmbda(T) / 2 * npo(tr(eps_e))**2 +
                 mu(T) * (npo(eigen_value_1)**2 + npo(eigen_value_2)**2))
    return psi_R_val


def g(phi):
    '''
    Stiffness modulation (degradation function) as a function of damage.

    Parameters
    ----------
    phi: FEniCS function
        The phase-field variable (damage)

    Returns
    -------
    g_val : FEniCS function
        The degradation of the stiffness at current damage level.
    '''
    eta = Constant(1e-6)
    g_val = (1 - phi) ** 2 + eta
    return g_val

def sigma(u, phi, P_v, T):
    '''
    Stress tensor.

    Parameters
    ----------
    u: FEniCS function
        Displacement field.

    phi: FEniCS function
        The phase-field variable (damage)

    P_v : FEniCS scalar function
        Pressure field.

    T : FEniCS scalar function
        Temperature field.

    Returns
    -------
    sigma_val : FEniCS function
        Stress tensor.

    Notes
    -----
    Following the nomenclature by Vicentini et al. 
    HAL Id: hal-04231075
    https://hal.sorbonne-universite.fr/hal-04231075

    The spectral decomposition follows the approach by Xue et al.
    https://doi.org/10.1016/j.cma.2021.114046
    https://github.com/tianjuxue/crack/blob/main/src/constitutive.py#L102
    '''
    eps_e = epsilon_elastic(u, P_v, T)
    epsilon = variable(eps_e)

    psi_D_partial = partial(psi_D, P_v=P_v, T=T)
    psi_D_val = psi_D_partial(epsilon)
    sigma_D = diff(psi_D_val, epsilon)

    psi_R_partial = partial(psi_R, P_v=P_v, T=T)
    psi_R_val = psi_R_partial(epsilon)
    sigma_R = diff(psi_R_val, epsilon)

    sigma_val = g(phi) * sigma_D + sigma_R
    return sigma_val


def H(u, P_v, T, H_n):
    '''
    History variable that enforces monotonically increase of the degradable
    elastic strain energy density.

    Parameters
    ----------
    u: FEniCS function
        Displacement field.

    P_v : FEniCS scalar function
        Pressure field.

    T : FEniCS scalar function
        Temperature field.

    H_n : FEniCS scalar function
        Previous value of elastic strain energy density.

    Returns
    -------
    H_val : FEniCS function
        The current value of elastic strain energy density.

    '''
    eps_e = epsilon_elastic(u, P_v, T)
    H_val = conditional(lt(H_n, psi_D(eps_e, P_v, T)), psi_D(eps_e, P_v, T), H_n)
    return H_val
        

def sigma_PS(u, phi, P_v, T):
    '''
    Magnitude of the first principal stress.

    Parameters
    ----------
    u: FEniCS function
        Displacement field.

    phi: FEniCS function
        The phase-field variable (damage)

    P_v : FEniCS scalar function
        Pressure field.

    T : FEniCS scalar function
        Temperature field.

    Returns
    -------
    sigma_PS_val : FEniCS function
        Magnitude of the first principal stress..

    Notes
    -----
    Follows the definition from:
    `https://fenicsproject.discourse.group/t/plotting-error-
    calculating-principal-stresses/3724`
    '''

    cauchy = project(sigma(u, phi, P_v, T), TS)
    sigma_x = cauchy[0,0] 
    sigma_y = cauchy[1, 1]
    tau_xy = cauchy[0, 1]
    sigma_PS_val = ((sigma_x + sigma_y)/2 +
                    sqrt(((sigma_x - sigma_y)/2)**2 + tau_xy**2))
    return sigma_PS_val
 
# Displacement problem Dirichlet boundary conditions (BC)
bc_b = DirichletBC(W.sub(1), Constant(0.), bottom)
bc_l = DirichletBC(W.sub(0), Constant(0.), left)
bcs_u = [bc_b, bc_l]

# Phase-field problem Dirichlet boundary conditions (BC)
bcs_phi = []

# Variational formulation of the displacement problem in residual form
def NonlinearDisplacementSolver():
    '''
    Constructor for the nonlinear displacement solver.

    Parameters
    ----------
    None.

    Returns
    -------
    solver_u : NonlinearVariationalSolver object
        The nonlinear solver for the displacement problem
    '''

    Res_u = inner(grad(v), sigma(u, phi_n, P_v, T)) * dx
    J_u = derivative(Res_u, u)
    problem_u = NonlinearVariationalProblem(Res_u, u, bcs_u, J_u, ffc_options)
    solver_u = NonlinearVariationalSolver(problem_u)
    prm_u = solver_u.parameters
    prm_u['newton_solver']['krylov_solver']['nonzero_initial_guess'] = True
    prm_u['newton_solver']['error_on_nonconvergence'] = False
    prm_u["newton_solver"]["maximum_iterations"] = 5
    prm_u["newton_solver"]["absolute_tolerance"] = 1E-7
    prm_u["newton_solver"]["relative_tolerance"] = 1E-10
    return solver_u


# Variational formulation of the phase-field problem in residual form
def LinearPhaseFieldSolver():
    '''
    Constructor for the Linear phase-field solver.

    Parameters
    ----------
    None.

    Returns
    -------
    solver_u : NonlinearVariationalSolver object
        The nonlinear solver for the displacement problem
    '''
    Res_phi = (Gc(T) * l_0 * inner(grad(phi_hat), grad(q_phi)) +
               ((Gc(T) / l_0) + 2.0 * H(u, P_v, T, Hold)) * inner(phi_hat, q_phi) -
               2.0 * H(u, P_v, T, Hold) * q_phi) * dx
    problem_phi = LinearVariationalProblem(lhs(Res_phi), rhs(Res_phi), phi,
                                           bcs_phi, ffc_options)
    solver_phi = LinearVariationalSolver(problem_phi)
    return solver_phi


solver_disp = NonlinearDisplacementSolver()
solver_phi = LinearPhaseFieldSolver()


###############################################################################
########################## 7. Temporal evolution loop #########################
###############################################################################

# IO Files - Postprocessing
time = []
convergence = []
filex = XDMFFile(dir_case + '/fields_' + case + '.xdmf')
filex.parameters['functions_share_mesh'] = True
filex.parameters['rewrite_function_mesh'] = False
filex.parameters["flush_output"] = True
freq_out = 1

file = open(dir_case + '/' + case + '_probes.csv', 'w')
writer = csv.writer(file, delimiter='\t')

# Loop parameters
t = 0           # Initial time
nt = 0          # Current number of time steps
tol = 1e-3      # Tolerance for the inner loop

print('    |- Starting:')

n_subs = 10

delta = Function(V)
deltaT, deltaP = delta.split(True)
P_v_theta = interpolate(P_0, V.sub(0).collapse())
P_v_n_i = interpolate(P_0, V.sub(0).collapse())

T_0 = Constant(298.15)
T_n = interpolate(T_0, V.sub(1).collapse())
T_theta = interpolate(T_0, V.sub(1).collapse())
T_n_i = interpolate(T_0, V.sub(1).collapse())

i_max = 50

# Starting time of simulation
startTime = datetime.now()
while t <= T_total:
    # 1. Updating heat-up curve
    T_flame.T = RABT(t)

    # 2. Solve non-linear thermal problem
    n_TH, conv_TH = solver_TH.solve()

    # 3. Update temperature values
    (_P, _T) = r.split(True)
    errs_max = []

    deltaT.vector()[:] = _T.vector() - T_n.vector()
    deltaP.vector()[:] = _P.vector() - P_v_n.vector()
    theta = 0

    dtheta = float(1 / n_subs)

    # 4. Solve the linear elastic phase field problem
    # (Enter the inner M-PF Loop)
    while theta < 1:
        # 4.1. Store the current time step
        dtheta_c = float(dtheta)
        # 4.2. Increment the current time step
        theta += dtheta_c
        # 4.3. Update the temperature in the inner loop
        T_theta.vector()[:] =  T_n.vector() + theta * deltaT.vector()
        P_v_theta.vector()[:] =  P_v_n.vector() + theta * deltaP.vector()

        # 4.4. Reset the inner iteration counter and the current maximum error
        i = 1
        err_max = 1

        # 4.5. Update the residuals
        solver_disp = NonlinearDisplacementSolver()
        solver_phi = LinearPhaseFieldSolver()

        # 4.6. Inner loop for staggered PF solver
        while (err_max > tol) and (i <= i_max):
            # 4.6.1. Solve the displacemenet problem
            solver_disp.solve()
            # 4.6.2. Solve the phase-field problem
            solver_phi.solve()
            # 4.6.3. Calculate the L-2 norm of the displacement error
            err_u = assemble(dot(u - u_n, u - u_n) * dx)**0.5
            # 4.6.4. Calculate the L-2 norm of the phase-field variable error
            err_phi = assemble((phi - phi_n)**2 * dx)**0.5
            # 4.6.5. Get the maximum from the errors
            err_max = max(err_u, err_phi)

            # 4.6.6. Update the current inner loop variables
            u_n.assign(u)
            phi_n.assign(phi)
            Hnew = project(H(u, P_v_theta, T_theta, Hold), V_H)
            Hold.assign(Hnew)
            i += 1

        # 4.9. Print if the maximum number of iterations was reached
        if i >= i_max:
            print(f'    |- !!! Maximum number of iterations ({i_max}) reached !!!')

        # 4.10. Calculate the potential energy increment
        Hnew = conditional(gt(phi, 0.1), 0, Hnew)
        dH = norm(project(Hnew - Hold, V_H).vector())

        errs_max.append(err_max)
        # 4.11. Save the data
        if (nt % freq_out == 0):
            phi.rename("Phase Field [-]", "phi")
            filex.write(phi, t - dt * (1 - theta))
            u.rename("Displacement [m]", "u")
            filex.write(u, t - dt * (1 - theta))
            Hold.rename("History Variable [-]", "H")
            filex.write(Hold, t - dt * (1 - theta))
            sig_PS = project(sigma_PS(u, phi, P_v_theta, T_theta), V_phi)
            sig_PS.rename('Maximum Principal Stress [MPa]', 'sig1')
            filex.write(sig_PS, t - dt * (1 - theta))
            _T.rename("Temperature [K]", "T")
            _P.rename("Pressure [Pa]", "P_v")
            filex.write(_T, t - dt * (1 - theta))
            filex.write(_P, t - dt * (1 - theta))

        # 5. Print the current step info
        print(f'    | ---------------------------------------------------- |')
        print(f'    | ---------------------------------------------------- |')
        msg = f'theta = {theta:.6f}s'
        print(f'    | {msg:^52} |')
        print(f'    |       i = {i:4g}             |    dtheta = {float(dtheta_c):.2e}    |')



    # 6. Update _n variables and print the information
    P_v_n.vector()[:] = _P.vector()
    T_n.vector()[:] = _T.vector()

    T_max = max(T_n.vector()[:]) - 273.15
    T_min = min(T_n.vector()[:]) - 273.15

    print((f'    |       Time [h] = {t/3600:.2f}' f'      |    % Complete = {t / T_total * 100:6.2f}  |\n'
           f'    |       T_min [°C] = {T_min:.2f}' f'   |    T_max [°C] = {T_max:7.2f} |'))

    # 7. Increase the current time step number
    nt += 1
    # 8. Increase the current time
    t += dt

# End loop over time steps
file.close()
time_delta = datetime.now() - startTime
print('\n    | Simulation time: ', str(time_delta))