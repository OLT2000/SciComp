import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

def cubic_f(x, c):
    return x**3 - x + c


#Natural Param Continuation
#aim to vary c between -2, 2
c_space = np.linspace(-2, 2, 101)
del_c = c_space[1] - c_space[0]
x_1 = 5
Xs = []
for c in c_space:
    solution = root(lambda x: cubic_f(x, c), x_1) #find root at c with previous x as initial guess
    x_1 = solution.x
    Xs.append(x_1)

plt.plot(c_space, Xs)
plt.title('Natural Parameter continuation')
plt.xlabel('Parameter [c]')
plt.ylabel('Equilibrium [x]')
plt.show()






