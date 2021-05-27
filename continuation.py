import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from types import FunctionType


def np_continuation(ode, init_vals, param_space):
    """

    :param ode: ODE to be solved
    :param init_vals: Initial known solution
    :param param_space: Range of values over which to vary the parameter, eg a numpy linspace
    :return:
    """
    if type(ode) != FunctionType:
        raise TypeError('Please check your ODE is a function')
    elif type(param_space) != np.ndarray and type(param_space) != tuple and type(param_space) != list:
        raise TypeError('Please check your parameter space is  a list of values to continue')
    sols = []
    for p in param_space:
        solution = root(lambda x: ode(x, p), init_vals) #find root at c with previous x as initial guess
        # if not solution.success:
        #     print('The root finder failed to converge')
        #     raise Exception(solution.message)
        init_vals = solution.x
        sols.append(init_vals)
    sols = np.array(sols)
    return sols


def pseudo_arc_length(ode, init_vectors, nsteps=1000):
    """
    :param ode: ODE to solved such that the fixed parameters are set inside the function, and the parameter to be varied is an argument
    :param init_vectors: Initial Vectors for pseudo arc length continuation, of the form [X, b] | b is the parameter being varied
    :param nsteps: Number of steps used to generate the parameter plot
    :return: An array of X and parameter values corresponding to the equilibrium as parameter changes
    """
    V_array = []
    if type(ode) != FunctionType:
        raise TypeError('Please check that your ODE is a function')
    elif len(init_vectors[0]) != len(init_vectors[1]):
        raise ValueError('Both initial vectors must be the same length')
    for v in init_vectors:
        if type(v) != list and type(v) != np.ndarray and type(v) != tuple:
            raise TypeError("Please check each of your initial vectors are of the type list, numpy.ndarray, or tuple [V0,  V1]")
        else:
            V_array.append(np.array(v))

    for vector in V_array:
        try:
            ode(vector[:-1], vector[-1])
        except ValueError:
            raise ValueError('Please check the dimensions of the initial conditions match those of the ODE.')
        except TypeError:
            raise TypeError('Please check  that the inputted vectors contain real numbers.')
    V0, V1 = V_array
    #create secant
    secant = V1 - V0
    #predict solution
    V_tilda = V1 + secant
    for n in range(nsteps):
        #define objective func to root
        def objective(v):
            b = v[-1]
            U = v[:-1]
            f_u = ode(U, b)
            dot_prod = np.dot((v - V_tilda), secant)
            return np.append(f_u, dot_prod)
        solution = root(objective, V_tilda)
        if not solution.success:
            print('The root finder failed to converge')
            raise Exception(solution.message)
        v_true = solution.x
        V_array.append(v_true)
        secant = V_array[-1] - V_array[-2]
        V_tilda = V_array[-1] + secant
    Final_V = np.array(V_array)
    Final_V = Final_V.transpose()
    Variables = np.array(Final_V[:-1])
    Params = Final_V[-1]
    return Variables.transpose(), Params

if __name__ == '__main__':
    c_space = np.linspace(2, 0, 1001)
    cubic_space = np.linspace(-2, 2, 1001)
    cubic0 = 5
    U0 = [1.41421356e+00, 1.39558408e-09]
    Xs = np_continuation(hopf_beta, U0, c_space)
    # Xs = Xs.transpose()

    plt.plot(c_space, Xs[0])
    plt.title('Natural Parameter continuation')
    plt.xlabel('Parameter [c]')
    plt.ylabel('Equilibrium [x]')
    plt.show()

    # U0 = [1.41421356e+00, 1.39558408e-09, 'z']
    # U1 = [1.39642400e+00, 1.26064267e-09]

    space = np.linspace(0, 2, 1000)
    c_x = np_continuation(hopf_beta, U0, space)

    plt.plot(space, c_x)
    plt.title('Pseudo Arc Length continuation')
    plt.xlabel('Parameter [c]')
    plt.ylabel('Equilibrium [x]')
    plt.show()
