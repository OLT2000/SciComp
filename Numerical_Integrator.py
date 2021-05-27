import numpy as np
import matplotlib.pyplot as plt
from numbers import Real
from types import FunctionType

def euler_step(x_i, h, t_i, fn):
    """
    Computes the next iterations of x and t using an Euler step function
    :param x_i: the initial x value(s) to the euler step
    :param h: step size
    :param t_i: initial time step
    :param fn: the function in x, t computing derivative
    :return: new x, and t values, one step forward in time.
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


def solve_to(x1, tspan, deltat_max, ode,  step_fn):
    """
    Numerically integrates a function from from t1 to t2, in a finite number of steps.
    During each step, temporary x values are computed, with the final one at time t2 is returned.
    :param x1: Initial position x values to solve from
    :param t1: Initial point in time to solve from
    :param t2: Upper limit in time to the integration
    :param deltat_max: Maximum time step to take between iterations
    :param ode: function to be integrated
    :param step_fn: choice of step function computing next x and t values
    :return: A single value for x at time t2
    """
    t1, t2 = tspan
    full_steps = int((t2 - t1) // deltat_max) #Whole number of steps available, governed by deltat_max. Converted to Int from float.
    for step in range(full_steps):
        x1, t1 = step_fn(x1, deltat_max, t1, ode)
    if t1 != t2: #Check if we have reached the final t2
        remainder = t2 - t1
        x1, tempt = step_fn(x1, remainder, t1, ode)
    return x1


def solve_ode(x1, timespan, nsteps, h, ode, solver=None):
    """
    :param x1: Initial point in x
    :param timespan: Timespan over which to integrate
    :param nsteps: Number of solution points generated
    :param h: Maximum step size
    :param ode: Function to be integrated
    :param solver: Method of solution, Euler for Euler's, and RK4 for RK4, if none, function will ask for input
    :return: N-D Array of solution points
    """
    if not isinstance(x1, Real):
        if not isinstance(x1, list) and not isinstance(x1, np.ndarray):
            raise TypeError('Check your initial point is of type int or list/array')
        else:
            for value in x1:
                if not isinstance(value, Real):
                    raise TypeError('All initial conditions must be real numbers')
    elif len(timespan) != 2:
        raise ValueError('Time span must be a 2 dimensional array. [T_start, T_end]')
    elif not isinstance(h, Real):
        raise TypeError('Stepsize, h, must be a real number')
    elif not isinstance(ode, FunctionType):
        raise TypeError('ODE must be a function')

    if solver == None: #if no solver was specified
        print("Please choose a Solver")
        solver = input("Euler or RK4")

    if solver == 'Euler':
        step_fn = euler_step
    elif solver == 'RK4':
        step_fn = rk4_step
    else:
        raise KeyError('Invalid solver. Please input Euler for Euler\'s method, or RK4 for Runge-Kutta')

    tstart, tend = timespan
    tarray = np.linspace(tstart, tend, nsteps)
    x_array = [x1]
    for i in range(nsteps-1):
        t1 = tarray[i]
        t2 = tarray[i+1]
        x_i_1 = solve_to(x_array[-1], [t1, t2], h, ode, step_fn)
        x_array.append(x_i_1)
    soln = np.array(x_array)
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
    ax1.set_title('Time plot: $x, v$ against $t$')
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


if __name__ == "__main__":
    # Code Testing
    # U0 = [0.1, 1]
    # t0 = 0
    # tf = 500
    # iters = 5000
    # r, tarr = solve_ode(U0, [t0, tf], iters, 0.01, hopf, 'RK4')
    # x = r[0]
    # y = r[1]
    # plot_solution(tarr, x, y)



    """
    Pred-Prey code
    # """
    # xy0 = [0.27015621, 0.27015621]
    # t0 = 0
    # tf = 100
    # span = [t0, tf]
    # iters = 10000
    # result, tarray = solve_ode(xy0, span, iters, 0.01, predatorprey, 'RK4')
    # x = result[0]
    # y = result[1]
    # plot_solution(tarray, x, y)
    # # print(tarray[5368] - tarray[5000], x[5368] - x[5000], y[5368]-y[5000])
    #
    #
    # """
    # 1D func code
    # """
    # x0 = 1
    # t0 = 0
    # tf = 10
    # iters = 200
    # # tarray = np.linspace(t0, tf, iters)
    # result, tarray = solve_ode(x0, [0, tf], iters, 0.01, function)
    # x = result
    # plot_solution1d(tarray, x)

    """
    Nd func code"""
    # x0 = [0.1, 1]
    # t0 = 0
    # tf = 100
    # iters = 1000
    # result, tarray = solve_ode(x0, [0, tf], iters, 0.01, predatorprey, 'RK4')
    # x = result[0]
    # v = result[1]
    # plot_solution(tarray, x, v)


    # Y0 = [-10, 3]
    # t0 = 0
    # tf = 10
    # h = 1
    # # tstart = perf_counter()
    # # rkvals, timespace = solve_ode(Y0, [t0, tf], 200,  h, test_ode, 'RK4')
    # # tend = perf_counter()
    # # print("Time elapsed for 200 points, h = 1", tend - tstart)
    # myvec = np.vectorize(test_true)
    # rk_e_array = []
    # eu_e_array = []
    # h_array = []
    # for idx in range(40):
    #     h = (1/2)**idx
    #     rkvals, tspace = solve_ode(Y0, [t0, tf], 200, h, test_ode, 'RK4')
    #     rkYs = rkvals[0]
    #     euvals, _ = solve_ode(Y0, [t0, tf], 200, h, test_ode, 'Euler')
    #     euYs = euvals[0]
    #     #Compute Error
    #     true_vals = myvec(tspace)
    #     rk_E = mse(true_vals, rkYs)
    #     eu_E = mse(true_vals, euYs)
    #     rk_e_array.append(rk_E)
    #     eu_e_array.append(eu_E)
    #     h_array.append(h)
    #
    # logvec = np.vectorize(lambda n: log(n, 4)) #Vectorise log2 function
    # eu_e_array = [logvec(x) for x in eu_e_array]
    # rk_e_array = [logvec(x) for x in rk_e_array]
    # h_array = [logvec(x) for x in h_array]
    #
    #
    # # ax1.set_title('Time series: $x$ against $t$')
    # plt.plot(h_array, eu_e_array, color='green', linewidth=2, label=r'Eulers')
    # plt.title('A plot to compare the errors for Euler\'s method and 4th order Runge-Kutta')
    # plt.plot(h_array, rk_e_array, label=r'RK4')
    # plt.legend()
    #
    #
    # plt.xlabel('log2[h]')
    # plt.ylabel('log2[error]')
    # plt.show()
