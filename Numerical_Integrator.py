import numpy as np
import matplotlib.pyplot as plt
from math import log10, log2
from math import e as exp


def euler_step(x_i, h, t_i, fn):
    """
    Computes the next iterations of x and t using an Euler step function
    :param x_i: the initial x value(s) to the euler step
    :param h: step size
    :param t_i: initial time step
    :param fn: the function in x, t computing derivative
    :return: new x, and t values, one step in forward.
    """
    deriv = np.array(fn(x_i, t_i))
    x_i_1 = x_i + h*deriv #perform euler step
    t_i_1 = t_i + h #iterate time
    return x_i_1, t_i_1


def rk4_step(x_i, h, t_i, fn):
    """
    Computes the next iterations of x and t using the fourth order Runge-Kutta step function
    :param x_i: the initial x value(s) to the euler step
    :param h: step size
    :param t_i: initial time step
    :param fn: the function in x, t to be integrated
    :return: new x, and t values, one step in forward.
    """
    #Compute RK4 integration values
    k1 = np.array(fn(x_i, t_i))
    k2 = np.array(fn(x_i + k1*(h/2), t_i + h/2))
    k3 = np.array(fn(x_i + k2*(h/2), t_i + h/2))
    k4 = np.array(fn(x_i + k3*h, t_i + h))
    x_i_1 = x_i + (h/6)*(k1 + 2*(k2+k3) + k4)
    t_i_1 = t_i + h
    return x_i_1, t_i_1


def solve_to(x1, tspan, deltat_max, fn,  step_fn):
    """
    Numerically integrates a function from from t1 to t2, in a finite number of steps.
    During each step, temporary x values are computed, with the final one at time t2 is returned.
    :param x1: Initial position x values to solve from
    :param t1: Initial point in time to solve from
    :param t2: Upper limit in time to the integration
    :param deltat_max: Maximum time step to take between iterations
    :param fn: function to be integrated
    :param step_fn: choice of step function computing next x and t values
    :return: A single value for x at time t2
    """
    t1, t2 = tspan
    full_steps = int((t2 - t1) // deltat_max) #Whole number of steps available, governed by deltat_max. Converted to Int from float.
    for step in range(full_steps):
        x1, t1 = step_fn(x1, deltat_max, t1, fn)
    if t1 != t2: #Check if we have reached the final t2
        remainder = t2 - t1
        x1, tempt = step_fn(x1, remainder, t1, fn)
    return x1


def solve_ode(x1, timespan, nsteps, h, fn, solver=None):
    """

    :param x1: Initial point in x
    :param timespan: Timespan over which to integrate
    :param nsteps: Number of solution points generated
    :param h: Maximum step size
    :param fn: Function to be integrated
    :param solver: Method of solution, Euler for Euler's, and RK4 for RK4, if none, function will ask for input
    :return: N-D Array of solution points
    """
    if solver == None: #if no solver was specified
        print("Please choose a Solver")
        solver = input("Euler or RK4")

    if solver == 'Euler':
        step_fn = euler_step
    elif solver == 'RK4':
        step_fn = rk4_step
    else:
        raise KeyError('Invalid solver. Please input Euler for EUler\'s method, or RK4 for Runge-Kutta')
    tstart, tend = timespan
    tarray = np.linspace(tstart, tend, nsteps)
    x_array = [x1]
    for i in range(nsteps-1):
        t1 = tarray[i]
        t2 = tarray[i+1]
        x_i_1 = solve_to(x_array[-1], [t1, t2], h, fn, step_fn)
        # print("x1", x_i_1)
        x_array.append(x_i_1)
    if type(x1) == int: #If we have a 1d system, ICs will be int not list
        soln = np.reshape(x_array, (-1, 1))
    else:
        soln = np.reshape(x_array, (-1, len(x1)))
    return soln.transpose(), tarray


def plot_solution1d(t, x):
    """Produce a figure with timeseries and phasespace plots"""

    # Create a figure with two plotting axes side by side:
    fig = plt.figure(figsize=(6, 3))
    ax1 = fig.add_axes([0.58, 0.15, 0.35, 0.7])

    # Timeseries plot
    ax1.set_title('Time series: $x$ against $t$')
    ax1.plot(t, x, color='green', linewidth=2, label=r'$x$')
    # ax1.set_yticks([-1, 0, 1])
    # ax1.set_xlabel(r'$t$')
    # ax1.set_xticks([0, np.pi, 2*np.pi, 3*np.pi])
    # ax1.set_xticklabels([r'$0$', r'$\pi$', r'$2\pi$', r'$3\pi$'])
    ax1.grid()
    ax1.legend()
    # Return the figure handle for showing/saving
    plt.show()
    return fig


def plot_solution(t, x, v):
    """Produce a figure with timeseries and phasespace plots"""

    # Create a figure with two plotting axes side by side:
    fig = plt.figure(figsize=(6, 3))
    ax1 = fig.add_axes([0.58, 0.15, 0.35, 0.7])
    ax2 = fig.add_axes([0.08, 0.15, 0.35, 0.7])

    # Timeseries plot
    ax1.set_title('Time series: $x, v$ against $t$')
    ax1.plot(t, x, color='green', linewidth=1, label=r'$x$')
    ax1.plot(t, v, color='blue', linewidth=1, label=r'$v$')
    # ax1.set_yticks([-1, 0, 1])
    ax1.set_xlabel(r'$t$')
    # ax1.set_xticks([0, np.pi, 2*np.pi, 3*np.pi])
    # ax1.set_xticklabels([r'$0$', r'$\pi$', r'$2\pi$', r'$3\pi$'])
    ax1.grid()
    ax1.legend()

    # Phasespace plot
    ax2.set_title('Phase space: $v$ against $x$')
    ax2.plot(x, v, linewidth=0.5, color='red')
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$v$', rotation=0)
    # ax2.set_xticks([-1, 0, 1])
    # ax2.set_yticks([-1, 0, 1])
    ax2.grid()

    # Return the figure handle for showing/saving
    plt.show()
    return fig


def predatorprey(X, t):
    # print(X)
    a = 1
    b = 0.25
    d = 0.1
    x, y = X
    dx = x*(1 - x) - (a*x*y)/(d+x)
    dy = b*y*(1-(y/x))

    return [dx, dy]


def hopf(U, t):
    u1, u2 = U
    beta = 2
    sigma = -1
    du1 = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    du2 = u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)
    return np.array([du1, du2])


# if __name__ == "__main__":

#
# # Code Testing
# U0 = [1, -1]
# t0 = 0
# tf = 7
# iters = 2000
# tarr = np.linspace(t0, tf, iters)
# r = solve_ode(U0, t0, tf, iters, 0.01, hopf, 'Euler')
# x = r[0]
# y = r[1]
# plot_solution(tarr, x, y)



"""
Pred-Prey code
# """
xy0 = [0.27015621, 0.27015621]
t0 = 0
tf = 4000
span = [t0, tf]
iters = 10000
result, tarray = solve_ode(xy0, span, iters, 0.01, predatorprey, 'RK4')
x = result[0]
y = result[1]
plot_solution(tarray, x, y)
# print(tarray[5368] - tarray[5000], x[5368] - x[5000], y[5368]-y[5000])


"""
1D func code
"""
# x0 = 1
# t0 = 0
# tf = 10
# iters = 200
# tarray = np.linspace(t0, tf, iters)
# result = solve_ode(x0, 0, tf, iters, 0.01, function)
# x = result
# plot_solution1d(tarray, x)

"""
Nd func code"""
# x0 = [0, 1]
# t0 = 0
# tf = 50
# iters = 1000
# tarray = np.linspace(t0, tf, iters)
# result = solve_ode(x0, 0, tf, iters, 0.01, f_shm)
# x = result[0]
# v = result[1]
# plot_solution(tarray, x, v)


# v = result[1]

# # # plt.plot(x, x, color='red')
# # plt.plot(x, v, color='green')
# # plt.show()
#
# # for iter in range(tf):
# #     h = (1/2)**iter
# #     rk_Val_array = solve_ode(x0, t0, tf, 200,  h, f_shm, rk4_step)
# #     eu_Val_array = solve_ode(x0, t0, tf, 200, h, f_shm, euler_step)
# #     print(rk_Val_array)
#
#     # eu_Final_val = eu_Val_array[-1]
#     # rk_Final_val = rk_Val_array[-1]
#     # rk_Error = abs(rk_Final_val - exp**tf)
#     # eu_Error = abs(eu_Final_val - exp**tf)
#     # rk_e_array.append(rk_Error)
#     # eu_e_array.append(eu_Error)
#     # h_array.append(h)
#
# # for i in range(len(h_array)):
# #     eu_e_array[i] = log2(eu_e_array[i])
# #     rk_e_array[i] = log2(rk_e_array[i])
# #     h_array[i] = log2(h_array[i])
#
# # fig = plt.figure(figsize=(6, 3))
# # ax1 = fig.add_axes([0.58, 0.15, 0.35, 0.7])
# #
# # # Timeseries plot
# # # ax1.set_title('Time series: $x$ against $t$')
# # ax1.plot(h_array, eu_e_array, color='green', linewidth=2, label=r'Eulers')
# # ax1.plot(h_array, rk_e_array, label=r'RK4')
# # ax1.grid()
# # ax1.legend()
# #
# #
# # plt.xlabel('log2[h]')
# # plt.ylabel('log2[error]')
# # plt.show()
# # x2 = solve_ode(x0, time, h, function, rk4_step)
# # print(x2)
# # plot_solution(time, x2)
# # plt.show()


