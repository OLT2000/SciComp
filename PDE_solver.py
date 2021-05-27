import numpy as np, types
import matplotlib.pyplot as plt
from math import pi, log2, e
import scipy as sp
import scipy.sparse
from scipy.sparse.linalg import spsolve
from types import FunctionType
from numbers import Real


def forward_euler_main(max_x, max_t, T, L, init_func, kappa, bcs, heat_source, bc_type=None):
    bc1 = bcs[0]
    bc2 = bcs[1]
    x = np.linspace(0, L, max_x+1)     # mesh points in space
    t = np.linspace(0, T, max_t+1)
    jarray = np.zeros(x.size)        # u at current time step
    jarray1 = np.zeros(x.size)
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)
    #Calculate initial conditions
    for i in range(0, max_x+1):
        jarray[i] = init_func(x[i])

    if lmbda >= 0.5:
        raise ValueError('Forward Euler is conditionally stable for lambda < 0.5, your lambda is:', lmbda)

    if bc_type == 'neumann':
        a = lmbda * np.ones(max_x+1)
        b = (1-2*lmbda)*np.ones(max_x+1)
        c = lmbda * np.ones(max_x+1)
        a[-2] = c[1] = 2*lmbda
        mtrx = np.array([a, b, c])
        pos = [-1, 0, 1]
        A_FE = sp.sparse.spdiags(mtrx, pos, max_x+1, max_x+1).todense()
        for j in range(max_t):
            pj = bc1(t[j])
            qj = bc2(t[j])
            #Matrix calculations
            b_array = np.zeros(jarray.size)
            b_array[0] = -pj
            b_array[-1] = qj
            jarray1 = np.dot(A_FE, jarray) + 2*lmbda*deltax*b_array + deltat*heat_source(x, t[j])

            # Save u_j at time t[j+1]
            jarray[:] = jarray1[:]
        return x, jarray
    elif bc_type == 'dirichlet':
        a = lmbda * np.ones(max_x-1)
        b = (1-2*lmbda)*np.ones(max_x-1)
        c = a
        mtrx = np.array([a, b, c])
        pos = [-1, 0, 1]
        A_FE = sp.sparse.spdiags(mtrx, pos, max_x-1, max_x-1).todense()
        for j in range(max_t):
            pj = bc1(t[j])
            qj = bc2(t[j])
            #Matrix calculations
            b_array = np.zeros(jarray[1:-1].size)
            b_array[0] = pj
            b_array[-1] = qj
            jarray1[1:-1] = np.dot(A_FE, jarray[1:-1]) + lmbda*b_array + deltat*heat_source(x[1:-1], t[j])

            # Set up BCs
            jarray1[0] = bc1(t[j])
            jarray1[max_x] = bc2(t[j])
            # Save u_j at time t[j+1]
            jarray[:] = jarray1[:]
        return x, jarray
    else:
        raise ValueError('Boundary conditions must be either dirichlet or neumann')


def FE_periodic(max_x, max_t, T, L, init_func):
    x = np.linspace(0, L, max_x+1)     # mesh points in space
    t = np.linspace(0, T, max_t+1)
    jarray = np.zeros(x.size)        # u at current time step
    jarray1 = np.zeros(x.size)
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)
    #Calculate initial conditions
    for i in range(0, max_x+1):
        jarray[i] = init_func(x[i])
    if lmbda >= 0.5:
        raise Exception('Forward Euler is conditionally stable for lambda < 0.5, your lambda is:', lmbda)
    a = lmbda * np.ones(max_x)
    b = (1-2*lmbda)*np.ones(max_x)
    c = a
    mtrx = np.array([a, b, c])
    pos = [-1, 0, 1]
    A_FE = sp.sparse.spdiags(mtrx, pos, max_x, max_x).todense()
    A_FE[0, max_x-1] = A_FE[max_x-1, 0] = lmbda
    for j in range(max_t):
        #Matrix calculations
        jarray1[:-1] = np.dot(A_FE, jarray[:-1])
        # Set up BCs
        jarray1[-1] = jarray[0]
        # Save u_j at time t[j+1]
        jarray[:] = jarray1[:]
    return x, jarray


def BE_periodic(max_x, max_t, T, L, init_func):
    x = np.linspace(0, L, max_x+1)     # mesh points in space
    t = np.linspace(0, T, max_t+1)
    jarray = np.zeros(x.size)        # u at current time step
    jarray1 = np.zeros(x.size)
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)
    #Calculate initial conditions
    for i in range(0, max_x+1):
        jarray[i] = init_func(x[i])
    a = -lmbda * np.ones(max_x)
    b = (1+2*lmbda)*np.ones(max_x)
    c = a
    mtrx = np.array([a, b, c])
    pos = [-1, 0, 1]
    A_BE = sp.sparse.spdiags(mtrx, pos, max_x, max_x).todense()
    A_BE[0, max_x-1] = A_BE[max_x-1, 0] = -lmbda
    for j in range(max_t):
        # pj = bc(t[j])
        #Matrix calculations
        jarray1[:-1] = sp.sparse.linalg.spsolve(A_BE, jarray[:-1])
        # Set up BCs
        jarray1[-1] = jarray[0]
        # Save u_j at time t[j+1]
        jarray[:] = jarray1[:]
    return x, jarray


def CN_periodic(max_x, max_t, T, L, init_func):

    x = np.linspace(0, L, max_x+1)     # mesh points in space
    t = np.linspace(0, T, max_t+1)
    jarray = np.zeros(x.size)        # u at current time step
    jarray1 = np.zeros(x.size)
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)
    #Calculate initial conditions
    for i in range(0, max_x+1):
        jarray[i] = init_func(x[i])
    a = -(lmbda/2) * np.ones(max_x)
    b = (1+lmbda)*np.ones(max_x)
    c = a
    mtrx = np.array([a, b, c])
    pos = [-1, 0, 1]
    A_BE = sp.sparse.spdiags(mtrx, pos, max_x, max_x).todense()
    A_BE[0, max_x-1] = A_BE[max_x-1, 0] = -lmbda/2

    a1 = (lmbda/2) * np.ones(max_x)
    b1 = (1-lmbda)*np.ones(max_x)
    c1 = a1
    mtrx1 = np.array([a1, b1, c1])
    pos = [-1, 0, 1]
    B_BE = sp.sparse.spdiags(mtrx1, pos, max_x, max_x).todense()
    B_BE[0, max_x-1] = B_BE[max_x-1, 0] = lmbda/2
    for j in range(max_t):
        # pj = bc(t[j])
        #Matrix calculations
        rhs = np.array(B_BE.dot(jarray[:-1]))
        jarray1[:-1] = sp.sparse.linalg.spsolve(A_BE, rhs[0])
        # Set up BCs
        jarray1[-1] = jarray[0]
        # Save u_j at time t[j+1]
        jarray[:] = jarray1[:]
    return x, jarray


def periodic_boundary(*args):
    discretisation = args[-1]
    if discretisation == 'forward':
        x, t = FE_periodic(*args[:-1])
    elif discretisation == 'backward':
        x, t = BE_periodic(*args[:-1])
    elif discretisation == 'cn':
        x, t = CN_periodic(*args[:-1])
    return x, t


def backwards_euler_main(max_x, max_t, T, L, init_func, kappa, bcs, heat_source, bc_type=None):
    bc1 = bcs[0]
    bc2 = bcs[1]
    x = np.linspace(0, L, max_x+1)     # mesh points in space
    t = np.linspace(0, T, max_t+1)
    jarray = np.zeros(x.size)        # u at current time step
    jarray1 = np.zeros(x.size)

    #Calculate initial conditions
    for i in range(0, max_x+1):
        jarray[i] = init_func(x[i])

    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)
    if bc_type == None:
        print("Please choose a boundary condition type")
        bc_type = input("dirichlet, or neumann")

    if bc_type == 'dirichlet':
        a = -lmbda * np.ones(max_x-1)
        b = (1+2*lmbda)*np.ones(max_x-1)
        c = -lmbda * np.ones(max_x - 1)
        mtrx = np.array([a, b, c])
        pos = [-1, 0, 1]
        A_BE = sp.sparse.spdiags(mtrx, pos, max_x-1, max_x-1).todense()

        for j in range(0, max_t):
            p_j1 = bc1(t[j+1])
            q_j1 = bc2(t[j+1])
            b_array = np.zeros(jarray[1:-1].size)
            b_array[0] = p_j1
            b_array[-1] = q_j1
            jarray1[1:-1] = scipy.sparse.linalg.spsolve(A_BE, jarray[1:-1]+lmbda*b_array+deltat*heat_source(x[1:-1], t[j]))

            #set boundary conditions
            jarray1[0] = p_j1
            jarray1[-1] = q_j1

            # Save u_j at time t[j+1]
            jarray[:] = jarray1[:]
        return x, jarray
    elif bc_type == 'neumann':
        a = -lmbda * np.ones(max_x+1)
        b = (1+2*lmbda)*np.ones(max_x+1)
        c = -lmbda * np.ones(max_x+1)
        a[-2] = c[1] = -2*lmbda

        mtrx = np.array([a, b, c])
        pos = [-1, 0, 1]
        A_BE = sp.sparse.spdiags(mtrx, pos, max_x+1, max_x+1).todense()
        for j in range(0, max_t):
            p_j1 = bc1(t[j+1])
            q_j1 = bc2(t[j+1])
            b_array = np.zeros(jarray.size)
            b_array[0] = -2*deltax*p_j1
            b_array[-1] = 2*deltax*q_j1
            jarray1 = scipy.sparse.linalg.spsolve(A_BE, jarray+lmbda*b_array+deltat*heat_source(x, t[j]))

            # Save u_j at time t[j+1]
            jarray[:] = jarray1[:]
        return x, jarray
    else:
        raise ValueError('Boundary conditions must be either dirichlet or neumann')


def cn_main(max_x, max_t, T, L, init_func, kappa, bcs, heat_source, bc_type=None):
    bc1 = bcs[0]
    bc2 = bcs[1]
    x = np.linspace(0, L, max_x+1)     # mesh points in space
    t = np.linspace(0, T, max_t+1)
    jarray = np.zeros(x.size)        # u at current time step
    jarray1 = np.zeros(x.size)

    #Calculate initial conditions
    for i in range(0, max_x+1):
        jarray[i] = init_func(x[i])

    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)
    if bc_type == None:
        print("Please choose a boundary condition type")
        bc_type = input("dirichlet, or neumann")
    if bc_type == 'dirichlet':
        a = -(lmbda/2) * np.ones(max_x-1)
        b = (1+lmbda)*np.ones(max_x-1)
        c = a
        b_b = (1-lmbda)*np.ones(max_x-1)

        mtrx_a = np.array([a, b, c])
        mtrx_b = np.array([-a, b_b, -c])
        pos = [-1, 0, 1]
        A_CN = sp.sparse.spdiags(mtrx_a, pos, max_x-1, max_x-1).todense()
        B_CN = sp.sparse.spdiags(mtrx_b, pos, max_x-1, max_x-1).todense()
        for i in range(0, max_x+1):
            jarray[i] = init_func(x[i]) #Calcs u_I at each x point
        # print(jarray.reshape(1, -1))
        for j in range(max_t):
            pj = bc1(t[j])
            pj1 = bc1(t[j+1])
            qj = bc2(t[j])
            qj1 = bc2(t[j+1])
            b_array = np.dot(B_CN, jarray[1:-1])
            bc_array = np.zeros(b_array.size)
            bc_array[0] = pj + pj1
            bc_array[-1] = qj + qj1
            b_array += (lmbda*bc_array + deltat*heat_source(x[1:-1], t[j]))
            #b_array is of the form matrix due to dot function, hence convert it to an array
            b_array = np.asarray(b_array)

            jarray1[1:-1] = scipy.sparse.linalg.spsolve(A_CN, b_array[0])

            #BCs
            jarray1[0] = pj
            jarray1[max_x] = qj

            # Save u_j at time t[j+1]
            jarray[:] = jarray1[:]
        return x, jarray

    elif bc_type == 'neumann':
        a = -(lmbda/2) * np.ones(max_x+1)
        a[-2] = -lmbda
        b = (1+lmbda)*np.ones(max_x+1)
        c = -(lmbda/2) * np.ones(max_x+1)
        c[1] = -lmbda

        a_b = (lmbda/2) * np.ones(max_x+1)
        a_b[-2] = lmbda
        b_b = (1-lmbda)*np.ones(max_x+1)
        c_b = (lmbda/2) * np.ones(max_x+1)
        c_b[1] = lmbda

        mtrx_a = np.array([a, b, c])
        mtrx_b = np.array([a_b, b_b, c_b])
        pos = [-1, 0, 1]
        A_CN = sp.sparse.spdiags(mtrx_a, pos, max_x+1, max_x+1).todense()
        B_CN = sp.sparse.spdiags(mtrx_b, pos, max_x+1, max_x+1).todense()
        for j in range(max_t):
            pj = bc1(t[j])
            pj1 = bc1(t[j+1])
            qj = bc2(t[j])
            qj1 = bc2(t[j+1])
            b_array = np.array(np.dot(B_CN, jarray))[0]
            bc_vec = np.zeros(jarray.size)
            bc_vec[1] = -(pj + pj1)
            bc_vec[-1] = qj + qj1
            b_array += (deltax*lmbda*bc_vec)
            jarray1 = sp.sparse.linalg.spsolve(A_CN, b_array+deltat*heat_source(x, t[j]))

            jarray[:] = jarray1[:]

        return x, jarray
    else:
        raise ValueError('Boundary conditions must be either dirichlet or neumann')


def finite_diff(init_func, max_x, max_t, T, L, kappa, heat_func, bcs = None, discretisation = None, bc_type = None):
    """

    :param init_func: Initial condition function at time = 0
    :param max_x: Number of X points in the solution space
    :param max_t: Number of time points in the solution space
    :param T: Upper time limit for the time domain
    :param L: Upper limit to the spatial domain
    :param kappa: Diffusion coefficient
    :param heat_func: Heat Source function. If zero or constant it must still be a constant. E.g lambda x: 0
    :param bcs: Boundary conditions. If None it implies periodic boundary conditions
    :param discretisation: Which finite difference method to use
    :param bc_type: Type of boundary condition. Neumann or Dirichlet
    :return: An array of the heat at time T, at each point in space x.
    """
    if not isinstance(init_func, FunctionType):
        raise TypeError('Please check your initial condition is of function type')
    elif not isinstance(heat_func, FunctionType):
        raise TypeError('Please ensure te heat source is a function, even if 0 or constant')
    elif not isinstance(max_x, int) or not isinstance(max_t, int):
        raise TypeError('Number of points in grid must be of type integer')
    elif not isinstance(T, Real) or not isinstance(L, Real):
        raise TypeError('Please ensure T and L are real numbers')

    if discretisation == None:
        print("Please choose a Discretisation")
        scheme = input("forward, backward or cn?")

    if discretisation == 'forward':
        scheme = forward_euler_main
    elif discretisation == 'backward':
        scheme = backwards_euler_main
    elif discretisation == 'cn':
        scheme = cn_main
    else:
        raise KeyError("Invalid discretisation. Please choose forward, backward, or cn")

    if not bcs:
        x, jarr = periodic_boundary(max_x, max_t, T, L, init_func, discretisation)
    else:
        if len(bcs) != 2:
            raise ValueError('Please check you have inputted 2 boundary conditions')
        elif not isinstance(bcs[0], FunctionType) or not isinstance(bcs[1], FunctionType):
            raise TypeError('Please check your boundary conditions are of type function')
        elif bc_type == None:
            bc_type == input('Please choose a boundary condition type neumann or dirichlet')

        if bc_type != 'neumann' and bc_type != 'dirichlet':
            raise KeyError('Invalid Boundary type.')
        x, jarr = scheme(max_x, max_t, T, L, init_func, kappa, bcs, heat_func, bc_type)
    return x, jarr


if __name__ == '__main__':
    def u_I(x):
        # initial temperature distribution
        y = np.sin(pi*x/L)
        # y = e**(-16*(x**2))
        # y = x*(1-x)
        return y


    def u_fx(x):
        return x*(1 - x)


    b1test = lambda t: 0
    b2test = lambda t: 0

    L = 1       # length of spatial domain
    T = 0.1

    mx = 30
    mt = 2000
    rhs_F = lambda x, t: 0

    # Set problem parameters/functions
    kappa = 1  # diffusion constant

    X, u_j = finite_diff(u_fx, mx, mt, T, L, kappa, rhs_F, bcs=[b1test, b2test],  bc_type='neumann', discretisation='backward')
    # X1, uj1 = finite_diff(, discretisation='forward')
    # X2, uj2 = finite_diff(u_fx, mx, mt, T, L, [b1test, b2test], bc_type='neumann', discretisation='cn')
    # #Plot the final result and exact solution
    # X3, uj3 = forward_euler_main(mx, mt, T, L, u_fx, [b1test, b2test], rhs_F, bc_type='dirichlet')
    # X4, uj4 = backwards_euler_main(mx, mt, T, L, u_fx, [b1test, b2test], rhs_F, bc_type='dirichlet')
    # X5, uj5 = cn_main(mx, mt, T, L, u_fx, [b1test, b2test], rhs_F, bc_type='dirichlet')

    xx = np.linspace(0,L,250)
    # plt.plot(xx,u_exact(xx,T),'b-',label='exact')

    # plt.plot(X3, uj3, 'bx',label='FE')
    # plt.plot(X4, uj4, 'ro', label='BE')
    # plt.plot(X5, uj5, 'gx', label='CN')

    plt.plot(X, u_j,'rx',label='BE')
    # plt.plot(X1, uj1,'bo',label='FE')
    # plt.plot(X2, uj2,'gx',label='CN')
    plt.xlabel('x')
    plt.ylabel('u(x, 0.1)')
    plt.legend()
    # plt.ylim([0, 0.25])
    plt.show()
