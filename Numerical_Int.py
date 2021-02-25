import numpy as np
import matplotlib.pyplot as plt
from math import log10, log2
from math import e as exp


def function(x, t):
    return x


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


def solve_ode(x1, tarray, h, fn, step_fn):
    x_array = [x1]
    nsols = len(tarray)
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
    plt.show()
    return fig

x0 = 1
t0 = 0
tf = 15
# h = 1
time = np.linspace(t0, tf, 11)
e_array = []
h_array = []

for iter in range(tf):
    h = (1/2)**iter
    Val_array = solve_ode(x0, time, h, function, rk4_step)
    Final_val = Val_array[-1]
    Error = abs(Final_val - exp**tf)
    e_array.append(Error)
    h_array.append(h)

for i in range(len(h_array)):
    e_array[i] = log2(e_array[i])
    h_array[i] = log2(h_array[i])

print(e_array)
print(h_array)
plot_solution(h_array, e_array)

# x2 = solve_ode(x0, time, h, function, rk4_step)
# print(x2)
# plot_solution(time, x2)
# plt.show()


