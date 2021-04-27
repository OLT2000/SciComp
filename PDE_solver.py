import numpy as np
import matplotlib.pyplot as plt
from math import pi, log2
import scipy as sp
import scipy.sparse
from scipy.sparse.linalg import spsolve
from sklearn.linear_model import LinearRegression

# import scipy.sparse as sp

# Set problem parameters/functions
kappa = 1.0   # diffusion constant
     # total time to solve for
def u_I(x):
    # initial temperature distribution
    y = np.sin(pi*x/L)
    return y


def u_exact(x, t):
    # the exact solution
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y


# u at next time step

def forwardeuler(max_x, max_t, T, L):
    # Set up the numerical environment variables
    x = np.linspace(0, L, max_x+1)     # mesh points in space
    t = np.linspace(0, T, max_t+1)
    jarray = np.zeros(x.size)        # u at current time step
    jarray1 = np.zeros(x.size)
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)    # mesh fourier number
    print("deltax=",deltax)
    print("deltat=",deltat)
    print("lambda=",lmbda)
    # Set initial condition
    for i in range(0, mx+1):
        jarray[i] = u_I(x[i]) #Calcs u_I at each x point
    print(jarray)
    # Solve the PDE: loop over all time points
    for j in range(0, max_t):
        # Forward Euler timestep at inner mesh points
        # PDE discretised at position x[i], time t[j]
        for i in range(1, max_x):
            jarray1[i] = jarray[i] + lmbda*(jarray[i-1] - 2*jarray[i] + jarray[i+1])

        # Boundary conditions
        jarray1[0] = 0; jarray1[mx] = 0

        # Save u_j at time t[j+1]
        # print("b4", jarray == jarray1)
        jarray[:] = jarray1[:]
        # print("aft", jarray == jarray1)

    return x, jarray


def fwdmatrix(max_x, max_t, T, L, pde):
    bc1 = lambda t: 0 #lambda function w.r.t t for boundary condition
    bc2 = lambda t: 0

    x = np.linspace(0, L, max_x+1)     # mesh points in space
    t = np.linspace(0, T, max_t+1)
    jarray = np.zeros(x.size)        # u at current time step
    jarray1 = np.zeros(x.size)
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)    # mesh fourier number
    a = lmbda * np.ones(max_x)
    b = (1-2*lmbda)*np.ones(max_x)
    c = a
    mtrx = np.array([a, b, c])
    pos = [-1, 0, 1]
    A_FE = sp.sparse.spdiags(mtrx, pos, max_x+1, max_x+1).todense()
    print("deltax=",deltax)
    print("deltat=",deltat)
    print("lambda=",lmbda)
    for i in range(0, max_x+1):
        jarray[i] = pde(x[i]) #Calcs u_I at each x point
    # print(jarray)
    for j in range(0, max_t):
        jarray1 = np.array(A_FE.dot(jarray))[0]

        jarray1[0] = bc1(t[j])
        jarray1[max_x] = bc2(t[j])
        # Save u_j at time t[j+1]
        jarray[:] = jarray1[:]
    return x, jarray


def backwardseuler(max_x, max_t, T, L, pde, verbose = False):
    x = np.linspace(0, L, max_x+1)     # mesh points in space
    t = np.linspace(0, T, max_t+1)
    jarray = np.zeros(x.size)        # u at current time step
    jarray1 = np.zeros(x.size)      # u at j+1 timestep
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)    # mesh fourier number
    a = -lmbda * np.ones(max_x+1)
    b = (1+2*lmbda)*np.ones(max_x+1)
    c = a
    mtrx = np.array([a, b, c])
    pos = [-1, 0, 1]
    A_BE = sp.sparse.spdiags(mtrx, pos, max_x+1, max_x+1).todense()
    if verbose == True:
        print("deltax =",deltax)
        print("deltat =",deltat)
        print("lambda =",lmbda)
    for i in range(0, max_x+1):
        jarray[i] = pde(x[i]) #Calcs u_I at each x point
    for j in range(0, max_t):
        # print()
        jarray1 = scipy.sparse.linalg.spsolve(A_BE, jarray)
        # jarray1 = np.linalg.solve(A_BE, jarray)

        #set boundary conditions
        jarray1[0] = 0
        jarray1[max_x] = 0

        # Save u_j at time t[j+1]
        jarray[:] = jarray1[:]
    return x, jarray


def cranknicholson(max_x, max_t, T, L, pde, verbose = False):
    x = np.linspace(0, L, max_x+1)     # mesh points in space
    t = np.linspace(0, T, max_t+1)
    jarray = np.zeros(x.size)        # u at current time step
    jarray1 = np.zeros(x.size)
    deltax = x[1] - x[0]            # gridspacing in x
    deltat = t[1] - t[0]            # gridspacing in t
    lmbda = kappa*deltat/(deltax**2)    # mesh fourier number
    a = -(lmbda/2) * np.ones(max_x+1)
    b = (1+lmbda)*np.ones(max_x+1)
    c = a
    b_b = (1-lmbda)*np.ones(max_x+1)

    mtrx_a = np.array([a, b, c])
    mtrx_b = np.array([-a, b_b, -c])
    pos = [-1, 0, 1]
    A_CN = sp.sparse.spdiags(mtrx_a, pos, max_x+1, max_x+1).todense()
    B_CN = sp.sparse.spdiags(mtrx_b, pos, max_x+1, max_x+1).todense()
    if verbose == True:
        print("deltax=",deltax)
        print("deltat=",deltat)
        print("lambda=",lmbda)
    for i in range(0, max_x+1):
        jarray[i] = pde(x[i]) #Calcs u_I at each x point
    # print(jarray)
    for j in range(max_t):
        b_array = np.array(B_CN.dot(jarray))[0]
        jarray1 = scipy.sparse.linalg.spsolve(A_CN, b_array)
        #BCs
        jarray1[0] = 0
        jarray1[max_x] = 0
        # Save u_j at time t[j+1]
        jarray[:] = jarray1[:]
    return x, jarray


def finite_diff(pde, max_x, max_t, T, L, discretisation = None):
    if discretisation == None:
        print("Please choose a Discretisation")
        discretisation = input("forward, backward or cn?")

    if discretisation == 'forward':
        discretisation = fwdmatrix
    elif discretisation == 'backward':
        discretisation = backwardseuler
    elif discretisation == 'cn':
        discretisation = cranknicholson
    else:
        print("Invalid discretisation\nPlease choose forward, backward, or cn")
        return -1
    x, jarr = discretisation(max_x, max_t, T, L, pde)
    return x, jarr


def get_slope(error, delta):
    X= np.array(delta)
    Y = np.array(error)
    model = LinearRegression()
    model.fit(X.reshape(-1,1), Y)
    return model.coef_


L = 1.0         # length of spatial domain
T = 0.5
# Set numerical parameters
mx = 10   # number of gridpoints in space
mt = 100   # number of gridpoints in time

# X, u_j = forwardeuler(mx, mt, T, L)
# X, u_j = backwardseuler(mx, mt, T, L, u_I)
# X, u_j = cranknicholson(mx, mt, T, L, u_I)
# X, u_j = fwdmatrix(mx, mt, T, L)
xx = np.linspace(0,L,250)
U_Exact = u_exact(xx, T)
Errors = []
Errors1 = []
delt = []
for n in range(25):
    tempdel = T/mt
    X, u_j = finite_diff(u_I, mx, mt, T, L, discretisation='cn')
    X1, u_j1 = finite_diff(u_I, mx, mt, T, L, discretisation='backward')
    Max_Error = abs(np.max(u_j) - np.max(U_Exact))
    MaxE1 = abs(np.max(u_j1) - np.max(U_Exact))
    Errors.append(Max_Error)
    Errors1.append(MaxE1)
    delt.append(tempdel)
    mt += 50



for e in range(len(Errors)):
    Errors[e] = log2(Errors[e])
    Errors1[e] = log2(Errors1[e])
    delt[e] = log2(delt[e])


plt.plot(delt, Errors)
plt.plot(delt, Errors1, 'r-')
plt.show()

slope = get_slope(Errors, delt)
slope1 = get_slope(Errors1, delt)
print("Slope for cn:", slope)
print("SLope for bwd:", slope1)

# print(ah)
#Plot the final result and exact solution
# plt.plot(X,u_j,'rx',label='num')
#
# plt.plot(xx,u_exact(xx,T),'b-',label='exact')
# plt.xlabel('X')
# plt.ylabel('u(x,0.5)')
# plt.legend(loc='upper right')
# plt.show()
