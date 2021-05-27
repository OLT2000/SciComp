import numpy as np
from Numerical_Integrator import solve_ode
from scipy.optimize import root
from numbers import Real
from types import FunctionType


def shooting(*args, verbose = False):
    """

    :param ode: The ODE to solved
    :param ics: the initial conditions to  estimate the shooting off of
    :param T_guess: An estimate for the time period of the limit cycle
    :param nsteps: number of steps in which integration is performed
    :param stepsize: stepsize for integration
    :param verbose: When True prints the information rather than just returning it
    :return: Corrected initial conditions that start on the limit cycle and the period of the limit cycle.
    """
    ode, ics, T_guess, nsteps, stepsize = args
    if type(ode) != FunctionType: #Check ODE is an actual function
        raise TypeError('Your ODE is not a function')

    elif type(ics) != list and type(ics) != int and type(ics) != np.ndarray and type(ics) != tuple:
            raise TypeError("Your initial conditions are of the form", type(ics), 'Please check your initial conditions are of the form Int (for a 1d system) or list/ndarray/tuple (for an Nd system)')

    try:
        ode(ics, T_guess)#Check to see if the initial conditions work with the ode
    except ValueError: #Dimension Mismatch
        print('Please check the dimensions of the initial conditions match those of the ODE.\nYou gave', len(ics), 'values')
        return -1
    except TypeError: #Type Mismatch
        print('Please check your initial conditions and time period are real numbers')
        return -1
    else:
        initial_guess = [x for x in ics]
        initial_guess.append(T_guess)
        for value in args[3:]:
            if not isinstance(value, Real):
                raise TypeError('All values must be a real number, however', value, 'is of type', type(value))
        def obj_fun(xyT): #Define objective function to be root found
            T = xyT[-1] #extract time period as final element
            xyT = np.array(xyT[:-1]) #extract variables
            soln, _ = solve_ode(xyT, [0, T], nsteps, stepsize, ode, solver='RK4')
            final_vals = soln.transpose()[-1]
            final_error = xyT - final_vals
            dxdt = ode(xyT, T)[0]
            obj = np.append(final_error, dxdt)
            return obj

        result = root(obj_fun, initial_guess)
        if not result.success:
            print('The root finder failed to converge')
            raise Exception(result.message)
        if verbose:
            print("The corrected initial values found:", result.x[:-1])
            print("The Period is:", result.x[-1])
        return result.x[:-1], result.x[-1]


if __name__ == '__main__':
    Period = 20
    # U0 = np.array(['egg', 0.8])
    U0 = 'egg'
    v0 = shooting(predatorprey, U0, Period, 500, 0.05, verbose=1)





