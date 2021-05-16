import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

def cubic_f(x, c):
    return x**3 - x + c


#Natural Param Continuation
#aim to vary c between -2, 2
c_space = np.linspace(-2, 2, 101)
c0 = c_space[0]
c1 = c_space[1]
del_c = c_space[1] - c_space[0]
xguess = 5

#set up v0, v1
u0 = root(lambda x: cubic_f(x, c0), xguess).x[0]
u1 = root(lambda x: cubic_f(x, c1), xguess).x[0]

v0 = np.array([c0, u0])
v1 = np.array([c1, u1])
#create secant
secant = v1 - v0

#Predict solution
v_tilda = v1 + secant

v_array = [v0, v1]
for n in range(100):
    #define objective func to root
    def objective(v):
        c, x = v
        f_v = cubic_f(x, c)
        PAL = np.dot((v - v_tilda), secant)
        return np.array([f_v, PAL])
    solution = root(objective, v_tilda)
    v_true = solution.x
    v_array.append(v_true)
    secant = v_array[-1] - v_array[-2]
    v_tilda = v_array[-1] + secant



v_array = np.array(v_array)
c_x = v_array.transpose()
Cs = c_x[0]
Xs = c_x[1]




plt.plot(Cs, Xs)
plt.title('Pseudo Arc Length continuation')
plt.xlabel('Parameter [c]')
plt.ylabel('Equilibrium [x]')
plt.show()






