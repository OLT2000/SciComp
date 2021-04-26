# %%
import numpy as np
from Numerical_Int import solve_ode
from scipy.integrate import odeint
from scipy.optimize import root
from scipy.optimize import fsolve
# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)





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


def F(timespace):
    x, v = s
    dxdt = v
    dvdt = -9.8
    dXdt = np.array([dxdt, dvdt])
    return dXdt


def predatorprey(X, t):
    a = 1
    b = 0.2
    d = 0.1
    # print(X)
    # print(X)
    x, y = X
    dx = x*(1 - x) - (a*x*y)/(d+x)
    dy = b*y*(1-(y/x))
    return [dx, dy]


def codetest(U, t):
    u1, u2, u3 = U
    beta = 1
    sigma = 1
    du1 = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    du2 = u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)
    du3 = -u3
    return np.array([du1, du2, du3])



def shooting(ode, ics, T_guess, nsteps):
    if type(ics) != list and type(ics) != int and type(ics) != np.ndarray:
        raise TypeError("Your initial conditions are of the form", type(ics), 'Please check your initial conditions are of the form Int (for a 1d system) or list/ndarray (for an Nd system)')
        # return -1
    initial_guess = [x for x in ics]
    initial_guess.append(T_guess)
    tarr = np.linspace(0, T_guess, nsteps)
    def obj_fun(xyT):
        T = xyT[-1]
        xyT = xyT[:-1]
        soln = solve_ode(xyT, 0, T, nsteps, 0.1, ode, solver='RK4')
        x = xyT[0]
        y = xyT[1]
        X = soln[0][-1]
        Y = soln[1][-1]
        dxdt = ode(xyT, T)[0]
        obj = np.array([x - X, y - Y, dxdt])
        return obj
    result = root(obj_fun, initial_guess)
    print("The corrected initial values found:", result.x[:-1])
    print("The Period is:", result.x[-1])
    return result.x


def hopf(U, t):
    u1, u2 = U
    beta = 1
    sigma = -1
    du1 = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    du2 = u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)
    return np.array([du1, du2])


Period = 6
U0 = np.array([1, -1])
# U0 = '2'
v0 = shooting(hopf, U0, Period, 1000)
# v_init = 10
# v0 = root(G_Func, xy0T)
# v0 = fsolve(G_Func, xy0T)
# v0 = fsolve(lambda ics: objective_new(ics, 50, [0, 5], 10, F), [-7, v_init])

print(v0)
