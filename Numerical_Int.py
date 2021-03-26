import numpy as np
import matplotlib.pyplot as plt
from math import log10, log2
from math import e as exp


def function(x, t):
    return x


def f_shm(X, t):
    x, v = X
    dxdt = v
    dvdt = -x
    dXdt = [dxdt, dvdt]
    return dXdt


def euler_step(x_i, h, t_i, fn):
    print("HERE", x_i, t_i, fn)
    print()
    deriv = np.array(fn(x_i, t_i))
    x_i_1 = x_i + h*deriv
    t_i_1 = t_i + h
    return x_i_1, t_i_1


def rk4_step(x_i, h, t_i, fn):
    k1 = np.array(fn(x_i, t_i))
    k2 = np.array(fn(x_i + k1*(h/2), t_i + h/2))
    k3 = np.array(fn(x_i + k2*(h/2), t_i + h/2))
    k4 = np.array(fn(x_i + k3*h, t_i + h))
    x_i_1 = x_i + (h/6)*(k1 + 2*(k2+k3) + k4)
    t_i_1 = t_i + h
    return x_i_1, t_i_1


def solve_to(x1, t1, t2, deltat_max, fn,  step_fn):
    Range = t2 - t1
    tempx = x1
    tempt = t1
    full_steps = Range // deltat_max
    for step in range(int(full_steps)):
        tempx, tempt = step_fn(tempx, deltat_max, tempt, fn)
        step += 1
    if tempt != t2:
        remainder = t2 - tempt
        tempx, tempt = step_fn(tempx, remainder, tempt, fn)

    return tempx


def solve_ode(x1, tstart, tend, nsteps, h, fn, solver=None):

    """
    :param x1: Initial point
    :param tstart: Start time
    :param tend: End time
    :param nsteps: Number of iterative steps taken
    :param h: Maximum step size
    :param fn: derivative function being integrated
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
        print("Invalid Solver")
        return -1
    tarray = np.linspace(tstart, tend, nsteps)
    nsols = len(tarray)
    x_array = [x1]
    for i in range(nsols-1):
        t1 = tarray[i]
        t2 = tarray[i+1]
        x_i_1 = solve_to(x_array[-1], t1, t2, h, fn, step_fn)
        # print("x1", x_i_1)
        x_array.append(x_i_1)
    if type(x1) == int: #If we have a 1d system, ICs will be int not list
        soln = np.reshape(x_array, (-1, 1))
    else:
        soln = np.reshape(x_array, (-1, len(x1)))
    return soln.transpose()


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
    ax1.plot(t, x, color='green', linewidth=2, label=r'$x$')
    ax1.plot(t, v, color='blue', linewidth=2, label=r'$v$')
    # ax1.set_yticks([-1, 0, 1])
    ax1.set_xlabel(r'$t$')
    # ax1.set_xticks([0, np.pi, 2*np.pi, 3*np.pi])
    # ax1.set_xticklabels([r'$0$', r'$\pi$', r'$2\pi$', r'$3\pi$'])
    ax1.grid()
    ax1.legend()

    # Phasespace plot
    ax2.set_title('Phase space: $v$ against $x$')
    ax2.plot(x, v, linewidth=2, color='red')
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
    b = 0.2
    d = 0.1
    x, y = X
    dx = x*(1 - x) - (a*x*y)/(d+x)
    dy = b*y*(1-(y/x))

    return [dx, dy]


# if __name__ == "__main__":



"""
Pred-Prey code
"""
# xy0 = [0.87, 0.3]
# t0 = 0
# tf = 150
# iters = 2500
# tarray = np.linspace(t0, tf, iters)
# result = solve_ode(xy0, t0, tf, iters, 0.01, predatorprey, 'RK4')
# x = result[0]
# y = result[1]
# plot_solution(tarray, x, y)



"""
1D func code
"""
# x0 = 1
# t0 = 0
# tf = 10
# iters = 200
# tarray = np.linspace(t0, tf, iters)
# result = solve_ode(x0, 0, tf, iters, 0.01, function, 'Euler')
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


