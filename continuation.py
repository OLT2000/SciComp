import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from Numerical_Shooting import shooting


def cubic_f(x, c):
    return x**3 - x + c


def hopf(U, beta):
    u1, u2 = U
    du1 = beta*u1 - u2 - u1*(u1**2 + u2**2)
    du2 = u1 + beta*u2 - u2*(u1**2 + u2**2)
    return np.array([du1, du2])


def pseudo_arc_length(ode, V0, V1):
    V_array = [V0, V1]
    #create secant
    secant = V1 - V0
    #predict solution
    V_tilda = V1 + secant

    for n in range(1000):
        #define objective func to root
        def objective(v):
            b = v[0]
            U = v[1:]
            f_u = ode(U, b)
            dot_prod = np.dot((v - V_tilda), secant)
            return np.append(dot_prod, f_u)
        solution = root(objective, V_tilda)
        v_true = solution.x
        V_array.append(v_true)
        secant = V_array[-1] - V_array[-2]
        V_tilda = V_array[-1] + secant
    Final_V = np.array(V_array)
    return Final_V.transpose()



U0 = np.array([2, 1.41421356e+00, 1.39558408e-09])
U1 = np.array([1.95, 1.39642400e+00, 1.26064267e-09])

c_x = pseudo_arc_length(hopf, U0, U1)

Cs = c_x[0]
Xs = c_x[1]
print(Cs)
print(Xs)

plt.plot(Cs, Xs)
plt.title('Pseudo Arc Length continuation')
plt.xlabel('Parameter [c]')
plt.ylabel('Equilibrium [x]')
plt.show()

# #Natural Param Continuation
# #aim to vary c between -2, 2
# c_space = np.linspace(-2, 2, 101)
# c0 = c_space[0]
# c1 = c_space[1]
# del_c = c_space[1] - c_space[0]
# xguess = 5
#
# #set up v0, v1
# u0 = root(lambda x: cubic_f(x, c0), xguess).x[0]
# u1 = root(lambda x: cubic_f(x, c1), xguess).x[0]
#
# v0 = np.array([c0, u0])
# v1 = np.array([c1, u1])
# #create secant
# secant = v1 - v0
#
# #Predict solution
# v_tilda = v1 + secant
#
# v_array = [v0, v1]
# for n in range(100):
#     #define objective func to root
#     def objective(v):
#         c, x = v
#         f_v = cubic_f(x, c)
#         PAL = np.dot((v - v_tilda), secant)
#         return np.array([f_v, PAL])
#     solution = root(objective, v_tilda)
#     v_true = solution.x
#     v_array.append(v_true)
#     secant = v_array[-1] - v_array[-2]
#     v_tilda = v_array[-1] + secant
#
# v_array = np.array(v_array)
# c_x = v_array.transpose()
# Cs = c_x[0]
# Xs = c_x[1]
#
# plt.plot(Cs, Xs)
# plt.title('Pseudo Arc Length continuation')
# plt.xlabel('Parameter [c]')
# plt.ylabel('Equilibrium [x]')
# plt.show()






