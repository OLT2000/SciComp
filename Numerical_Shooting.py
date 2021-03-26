# %%
import numpy as np
from Numerical_Int import solve_ode
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


# %%
def predatorprey(t, X):
    print(X)
    x, y = X
    dx = x*(1 - x) - (a*x*y)/(d+x)
    dy = b*y*(1-(y/x))

    return [dx, dy]


def f_shm(X, t):
    x, v = X
    dxdt = v
    dvdt = -x
    dXdt = [dxdt, dvdt]
    return dXdt


# f_launch = lambda t, s: np.dot(np.array([[0,1],[0,-9.8/s[1]]]),s)
def f_launch(X, t):
    x, v = X
    dxdt = v
    dvdt = -9.8
    dXdt = [dxdt, dvdt]
    return dXdt


# def F(t, s):
#     A = np.array([[0, 1], [0, -9.8/s[1]]])
#     return np.dot(A, s)


def F(t, s):
    x, v = s
    dxdt = v
    dvdt = -9.8
    dXdt = np.array([dxdt, dvdt])
    return dXdt



t_eval = np.linspace(0, 5, 10)



def objective(x0, v0):
    sol = solve_ivp(F, [0, 5], [x0, v0], t_eval = t_eval)
    y = sol.y[0]
    return y[-1] - 50


def objfunc(x0, v0, t, func):
    ics = np.array([x0, v0])
    soln = solve_ode(ics, 0, t, 10, 0.01, func)
    y = soln[0]
    return y - 50


x0 = 0
v_init = 10
# v0 = fsolve(lambda v: objective(x0, v), 10)
v0 = fsolve(lambda v: objfunc(x0, v, 5, F), v_init)

print(v0)


# a = 1.0
# b = 0.2
# d = 0.1
#
# p = (a, b, d)  # Parameterspace
#
# y0 = [1, 1]  # Initial conditions
# x0 = [0, 1]
# t = np.linspace(0, 50, 200)
# # t_span = (0, t[-1])
# # t = np.arange(0.0, 40.0, 0.01)
#
# result = solve_ode(y0, 0, t[-1], 200, 0.1, predatorprey)
#
# # result = solve_ivp(predatorprey, t_span, y0, args=p)
# # print(result)
# Xs = result[0]
# Ys = result[1]
# T = np.arange(0, len(Xs))
#
#
# fig = plt.figure(figsize=(6, 3))
# ax1 = fig.add_axes([0.58, 0.15, 0.35, 0.7])
#
# # Timeseries plot
# # ax1.set_title('Time series: $x$ against $t$')
# ax1.plot(T, Xs, color='green', linewidth=2, label=r'X')
# ax1.plot(T, Ys, label=r'Y')
# ax1.grid()
# ax1.legend()
#
# plt.show()


