from matplotlib import pyplot as plt
from math import cos, sin, pi
import numpy as np
from Numerical_Integrator import solve_ode


def get_error(predicted, true, min_tol = 1e-10):
    accurate = True
    tolerance = 1
    while accurate and tolerance > min_tol:
        accurate = np.allclose(predicted, true, atol=tolerance)
        tolerance *= 1/10
    return tolerance

#Numerical Integration
#Define Our ODE
def example_ode(Y, t):
    y, v = Y
    dy = v
    dv = -16*y
    return [dy, dv]


#Define our true solution
def test_true(t):
    sol = -10*cos(4*t) + (3/4)*sin(4*t)
    return sol


myVectorFunc = np.vectorize(test_true)

Y0 = [-10, 3] #Define Initial conditions
timeframe = [0, 10] #Define Timeframe over which to integrate
stepsize = 0.01 #Define stepsize
soln_points = 100

#Use ODE solver
Y_solutions_rk4, Timespace = solve_ode(Y0, timeframe, soln_points, stepsize, example_ode, 'RK4')
Y_sol = Y_solutions_rk4[0]
V_sol = Y_solutions_rk4[1]

Y_True = myVectorFunc(Timespace) #Vectorize true solution function

plt.plot(Timespace, Y_sol, 'rx',label='Num')
plt.plot(Timespace, Y_True, label='True')
plt.xlabel('Time [t]')
plt.ylabel('Y')
plt.legend(loc='upper right')
plt.show()

error = get_error(Y_sol, Y_True)

print("The error for the Runge-Kuuta scheme is", error, '\n')


#Limit Cycles and Shooting
from Numerical_Shooting import shooting
#Define our ODE
def hopf(U, t):
    u1, u2 = U
    beta = 1
    sigma = -1
    du1 = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    du2 = u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)
    return [du1, du2]


def exact_hopf(t, beta = 1):
    u1 = (beta**0.5) * cos(t)
    u2 = (beta**0.5) * sin(t)
    return u1, u2

myVecHopf = np.vectorize(exact_hopf)

#Define initial conditions and approximate period (possibly found using a plot from solve_ode)
Period_estimate = 6 #Period
U0 = [0.1, 0.2] #initial points
#Define Parameters for ode solver
num_steps = 100
stepsize = 0.01
correct_points, period = shooting(hopf, U0, Period_estimate, num_steps, stepsize, verbose=1) #Verbose prints the solution
#Generate solution points to test limit cycle
limit_cycle, timeframe = solve_ode(correct_points, [0, period], num_steps, stepsize, hopf, 'RK4')
u1 = limit_cycle[0]
u2 = limit_cycle[1]

u1true, u2true = myVecHopf(timeframe)

plt.plot(timeframe, u1, 'rx', label='u1 Num')
plt.plot(timeframe, u2,  'bx', label='u2 Num')
plt.plot(timeframe, u1true, label='u1 true')
plt.plot(timeframe, u2true, label='u2 true')
plt.xlabel('time [t]')
plt.ylabel('U')
plt.legend(loc = 'lower right')
plt.show()

error_shoot = get_error(u1, u1true)
print("The shooting error is approximately", error_shoot, '\n')


#Parameter Continuation
from continuation import np_continuation, pseudo_arc_length
#Define our function
#My continuation can only handle one parameter at a time. All the others must be hardcoded as follows
def cubic_equation(x, d):
    a = 1
    b = 0
    c = -1
    return a*x**3 + b*x**2 + c*x + d


#Define our parameter space
param_space = np.linspace(-2, 2, 1001)
#Set our initial condition (that is a pre-existing solution)
f0 = 1.52138
np_solution = np_continuation(cubic_equation, f0, param_space)

#Set up init vals for pseudo (Once again must be known solutions)
Init_Vector1 = [1.52138, -2] #[Initial equilibria, parameter]
Init_Vector2 = [1.51291, -1.95]
vecspace = [Init_Vector1,  Init_Vector2]
iterations = 100
pal_solution, pal_param_space = pseudo_arc_length(cubic_equation, vecspace, iterations)

plt.plot(param_space, np_solution, label='np')
plt.plot(pal_param_space, pal_solution, label='PAL')
plt.legend(loc = 'lower right')
plt.xlabel('Parameter')
plt.ylabel('Equilibrium')
plt.show()




#PDE problems
from PDE_solver import finite_diff

#Define our Initial Condition
def u_initial(x, L = 1):
    y = np.sin(pi*x/L)
    return y

#Define our exact solution
def u_exact(x, t, kappa = 1, L = 1):
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y


#Define our boundary conditions and heat source as functions (even if zero or constant)
boundary1 = lambda t: 0
boundary2  = lambda t:0
boundaries = [boundary1, boundary2]

heatsource = lambda x, t: 0 #Must vary in space and time

#Set up Mesh Parameters and diffusion coefficient
xpoints = 20
tpoints = 1000
diff_coeff = 1
#Define our limits
t_upper = 1
x_upper = 1


X1, u_fe = finite_diff(u_initial, xpoints, tpoints, t_upper, x_upper, diff_coeff, heatsource, bcs=boundaries,  bc_type='dirichlet', discretisation='forward')
X2, u_be = finite_diff(u_initial, xpoints, tpoints, t_upper, x_upper, diff_coeff, heatsource, bcs=boundaries,  bc_type='dirichlet', discretisation='backward')
X3, u_cn = finite_diff(u_initial, xpoints, tpoints, t_upper, x_upper, diff_coeff, heatsource, bcs=boundaries,  bc_type='dirichlet', discretisation='cn')

xx = np.linspace(0, x_upper, 21)
true_u = u_exact(xx, t_upper)

plt.plot(X1, u_fe, 'bo', label='FE')
plt.plot(X2, u_be, 'rx', label='BE')
plt.plot(X3, u_cn, 'yx', label='CN')
plt.plot(xx, true_u, label='true')
plt.show()
error_fe = get_error(u_fe, true_u)
error_be = get_error(u_be, true_u)
error_cn = get_error(u_cn, true_u)
print("The error for forward euler is", error_fe)
print("The error for backwards euler is", error_be)
print("The error for crank-nicolson is", error_cn)

