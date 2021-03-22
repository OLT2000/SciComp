import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def testfunc(t, state, a, b, d):
    x, y = state

    dx = x*(1 - x)- (a*x*y)/(d+x)
    dy = b*y*(1-(y/x))

    return [dx, dy]

a = 1.0
b = 0.4
d = 0.1

p = (a, b, d)  # Parameterspace

y0 = [10, 5]  # Initial conditions
t = np.linspace(0, 50, 300)
t_span = (0.0, t[-1])
# t = np.arange(0.0, 40.0, 0.01)

result = solve_ivp(testfunc, t_span, y0, args=p)
# print(result)
Xs = result.y[0]
Ys = result.y[1]
T = np.arange(0, len(Xs))


fig = plt.figure(figsize=(6, 3))
ax1 = fig.add_axes([0.58, 0.15, 0.35, 0.7])

# Timeseries plot
# ax1.set_title('Time series: $x$ against $t$')
ax1.plot(T, Xs, color='green', linewidth=2, label=r'X')
ax1.plot(T, Ys, label=r'Y')
ax1.grid()
ax1.legend()

plt.show()


