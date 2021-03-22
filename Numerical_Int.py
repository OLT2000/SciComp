import numpy as np
import matplotlib.pyplot as plt
from math import log10, log2
from math import e as exp


def main(x_init, t_init, tend, nsteps, h, fn):
    time = np.linspace(t_init, tend, nsteps)
    rk_Val_array = solve_ode(x_init, t_init, tend, nsteps,  h, fn, rk4_step)
    eu_Val_array = solve_ode(x_init, t_init, tend, nsteps, h, fn, euler_step)
    eu = get_Error(eu_Val_array, time)
    rk = get_Error(rk_Val_array, time)


def get_Error(valarray, tarray):
    solnarray = np.array(tarray)
    soln = exp**(solnarray)
    error_array = abs(valarray-tarray)
    print(error_array)


def function(x, t):
    return x


def f_shm(X, t):
    x, v = X
    dxdt = v
    dvdt = -x
    dXdt = [dxdt, dvdt]
    return dXdt


def euler_step(x_i, h, t_i, fn):
    x_i_1 = x_i + h*fn(x_i,  t_i)
    t_i_1 = t_i + h
    return x_i_1, t_i_1


def rk4_step(x_i, h, t_i, fn):
    k1 = fn(x_i, t_i)
    k2 = fn(x_i + k1*(h/2), t_i + h/2)
    k3 = fn(x_i + k2*(h/2), t_i + h/2)
    k4 = fn(x_i + k3*h, t_i + h)
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


def solve_ode(x1, tstart, tend, nsteps, h, fn, step_fn):
    tarray = np.linspace(tstart, tend, nsteps)
    nsols = len(tarray)
    x_array = [x1]
    # x_array = np.array([vars, nsols])
    for i in range(nsols-1):
        t1 = tarray[i]
        t2 = tarray[i+1]
        x_i_1 = solve_to(x_array[-1], t1, t2, h, fn, step_fn)
        x_array.append(x_i_1)
    return x_array


def plot_solution(t, x):
    """Produce a figure with timeseries and phasespace plots"""

    # Create a figure with two plotting axes side by side:
    fig = plt.figure(figsize=(6, 3))
    ax1 = fig.add_axes([0.58, 0.15, 0.35, 0.7])

    # Timeseries plot
    # ax1.set_title('Time series: $x$ against $t$')
    ax1.plot(t, x, color='green', linewidth=2, label=r'$x$')
    ax1.grid()
    ax1.legend()

    # Return the figure handle for showing/saving
    # plt.show()
    return fig


x0 = 1
t0 = 0
tf = 15

rk_e_array = []
h_array = []
eu_e_array = []

for iter in range(tf):
    h = (1/2)**iter
    rk_Val_array = solve_ode(x0, t0, tf, 200,  h, function, rk4_step)
    eu_Val_array = solve_ode(x0, t0, tf, 200, h, function, euler_step)
    eu_Final_val = eu_Val_array[-1]
    rk_Final_val = rk_Val_array[-1]
    rk_Error = abs(rk_Final_val - exp**tf)
    eu_Error = abs(eu_Final_val - exp**tf)
    rk_e_array.append(rk_Error)
    eu_e_array.append(eu_Error)
    h_array.append(h)

for i in range(len(h_array)):
    eu_e_array[i] = log2(eu_e_array[i])
    rk_e_array[i] = log2(rk_e_array[i])
    h_array[i] = log2(h_array[i])

fig = plt.figure(figsize=(6, 3))
ax1 = fig.add_axes([0.58, 0.15, 0.35, 0.7])

# Timeseries plot
# ax1.set_title('Time series: $x$ against $t$')
ax1.plot(h_array, eu_e_array, color='green', linewidth=2, label=r'Eulers')
ax1.plot(h_array, rk_e_array, label=r'RK4')
ax1.grid()
ax1.legend()


plt.xlabel('log2[h]')
plt.ylabel('log2[error]')
plt.show()
# x2 = solve_ode(x0, time, h, function, rk4_step)
# print(x2)
# plot_solution(time, x2)
# plt.show()


