# %%
import numpy as np
from Numerical_Int import solve_ode
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import root
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


def predatorprey(X, t):
    a = 1
    b = 0.2
    d = 0.1
    x, y = X
    dx = x*(1 - x) - (a*x*y)/(d+x)
    dy = b*y*(1-(y/x))
    return [dx, dy]


def G_Func(xyT):
    # x, y = xy0
    x, y, T = xyT
    tarray = np.linspace(0, T, 1000)
    soln = odeint(predatorprey, [x, y], tarray).transpose()
    # print(x, y)
    # print(soln, T)
    X = soln[0][-1]
    Y = soln[1][-1]
    dxdt = x*(1 - x) - (1*x*y)/(0.1+x)
    G = np.array([x - X, y - Y, dxdt])
    return G


def objfun(x, y, prd):
    x, y = xy
    tarray = np.linspace(0, prd, 1000)
    soln = odeint(predatorprey, [x, y], tarray).transpose()
    X = soln[0][-1]
    Y = soln[1][-1]
    dxdt = x*(1 - x) - (1*x*y)/(0.1+x)
    G = np.array([x - X, y - Y, dxdt])
    return G


def objective(x0, v0):
    # print(x0, v0)
    t_eval = np.linspace(0, 5, 10)
    sol = solve_ivp(F, [0, 5], [x0, v0], t_eval = t_eval)
    y = sol.y[0]
    # print(sol.y[-1])
    return y[-1] - 50


def objective2(x0, y0):
    soln = solve_ivp(predatorprey, [0, 100], [x0, y0])
    x = sol.y[0]
    y = sol.y[1]




def objective_new(ics, bcs, tspan, steps, func):
    t_eval = np.linspace(tspan[0], tspan[1], steps)
    sol = solve_ivp(F, tspan, ics, t_eval=t_eval)
    result = sol.y
    resultarray = []
    for r in result:
        resultarray.append(r[-1])
    finalval = np.array(resultarray)
    finalval -= np.array(bcs)
    return finalval


# xy0 = [0.2, 0.3]
Period = 20
x0 = 0.2
y0 = 0.3
xy0T = [0.2, 0.3, 20]
# v_init = 10
v0 = root(G_Func, xy0T)
# v0 = fsolve(lambda ics: objective_new(ics, 50, [0, 5], 10, F), [-7, v_init])

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


