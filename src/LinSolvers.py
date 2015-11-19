'''
Author: PY Chuang
Python version: Python 2.7
Date: March 2015
'''


import numpy
from scipy import sparse
from scipy import weave

def u_exact1(x, y, n):
    '''
    exact solution for the problem Dirichlet BC in the assignment.
    '''
    temp = 2. * numpy.pi * n
    return  numpy.sin(temp * x) * numpy.sin(temp * y)
    #return numpy.exp(x) * numpy.exp(-2 * y)

def u_exact2(x, y, n):
    '''
    exact solution for the problem Neumann BC in the assignment.
    '''
    temp = 2. * numpy.pi * n
    return  numpy.cos(temp * x) * numpy.cos(temp * y)



def source_term_1(x, y, n):
    '''
    source term f = - 8 n^2 pi^2 sin(2 pi n x) sin(2 pi n y)
    '''
    c = 2 * numpy.pi * n
    return - 2 * c * c * numpy.sin(c * x) * numpy.sin(c * y)
    #return 5 * numpy.exp(x) * numpy.exp(-2 * y)

def source_term_2(x, y, n):
    '''
    source term f = - 8 n^2 pi^2 cos(2 pi n x) cos(2 pi n y)
    '''
    c = 2 * numpy.pi * n
    return - 2 * c * c * numpy.cos(c * x) * numpy.cos(c * y)

def set_DirichletBC(X, Y, u, fuc, n):
    '''
    Set up Dirichlet BC according to the exact solution.
    '''
    u[:, 0] = fuc(X[:, 0], Y[:, 0], n)
    u[:, -1] = fuc(X[:, -1], Y[:, -1], n)
    u[0, :] = fuc(X[0, :], Y[0, :], n)
    u[-1, :] = fuc(X[-1, :], Y[-1, :], n)
    return u

def set_NeumannBC(u, refP):
    '''
    Adiabatic BC. refP is always at u[0, 1]
    '''
    u[0, 1:-1] = u[1, 1:-1]
    u[-1, 1:-1] = u[-2, 1:-1]
    u[1:-1, 0] = u[1:-1, 1]
    u[1:-1, -1] = u[1:-1, -2]
    u[0, 1] = refP

def TrivialFunc(*trivial):
    pass

def residual(u, dx, dy, f):
    '''
    Calculate residual.
    '''
    r = numpy.zeros_like(u)
    r[1:-1, 1:-1] = f[1:-1, 1:-1] - \
            (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dy / dy - \
            (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dx / dx
    return r


def L2norm(x):
    '''
    Return L2-norm of a vector x.
    '''
    return numpy.sqrt(numpy.sum(x**2))


def JacobiMethod(Nx, Ny, u, f, dx, dy, tol, BCtype, refP=None, **opt):
    '''
    Jacobi iteration of the Poisson equation. Only for dx = dy = constant.
    '''
    assert dx == dy

    BCHandle = {'N': set_NeumannBC, 'D': TrivialFunc}

    dx2 = dx * dx
    err = 1000.
    itr_count = 0
    while (err > tol and itr_count < int(1e8)):
        
        itr_count += 1
        
        un = u.copy()

        BCHandle[BCtype](u, refP)

        expr = ("u[1:-1, 1:-1] = 0.25 * (un[1:-1, 2:] + un[1:-1, :-2] + "
                "un[2:, 1:-1] + un[:-2, 1:-1] - dx2 * f[1:-1, 1:-1])")
        weave.blitz(expr, check_size=0)

        err = L2norm(u[1:-1, 1:-1] - un[1:-1, 1:-1]) 
        print itr_count, err, (u-un).max(), (u-un).min()


    BCHandle[BCtype](u, refP)

    return u, itr_count


def SOR(Nx, Ny, u, f, dx, dy, tol, BCtype, refP=None, omg=1.9):
    '''
    SOR for 2D Poisson equation. Only for dx = dy = constant.
    '''
    assert dx == dy

    BCHandle = {'N': set_NeumannBC, 'D': TrivialFunc}

    dx2 = dx * dx
    err = 1000.
    itr_count = 0
    while (err > tol and itr_count < int(1e8)):

        itr_count += 1

        un = u.copy()

        BCHandle[BCtype](u, refP)

        expr = ("u[1:-1, 1:-1] = 0.25 * omg * (u[1:-1, 2:] + u[1:-1, :-2] + "
                "u[2:, 1:-1] + u[:-2, 1:-1] - dx2 * f[1:-1, 1:-1]) + " 
                "(1 - omg) * u[1:-1, 1:-1]")
        weave.blitz(expr, check_size=0)

        err = L2norm(u[1:-1, 1:-1] - un[1:-1, 1:-1]) 
        if itr_count % 10000 == 0:
            print itr_count, err, (u-un).max(), (u-un).min()


    BCHandle[BCtype](u, refP)

    return u, itr_count


def CG(Nx, Ny, u, f, dx, dy, tol, BCtype, refP=None, **opt):
    '''
    Conjugate gradient method. A general linear solver for problem Ax=b.
    '''
    assert dx == dy

    BCHandle = {'N': set_NeumannBC, 'D': TrivialFunc}
   
    A = sparse.diags([-4]*Nx*Ny, 0)
    A += sparse.diags([1]*(Nx-1)+([0]+[1]*(Nx-1))*(Ny-1), 1)
    A += sparse.diags([1]*(Nx-1)+([0]+[1]*(Nx-1))*(Ny-1), -1)    
    A += sparse.diags([1]*(Nx*Ny-Nx), Nx)
    A += sparse.diags([1]*(Nx*Ny-Nx), -Nx)   

    if BCtype == 'D':
        b = dx * dx * f[1:-1, 1:-1].reshape(Nx * Ny) 
        print "b = \n"
        print b
        b[:Nx] += -u[0, 1:-1]
        b[::Nx] += -u[1:-1, 0]
        b[Nx-1::Nx] += -u[1:-1, -1]
        b[-Nx:] += -u[-1, 1:-1]
    elif BCtype == 'N':
        b = dx * dx * f[1:-1, 1:-1].reshape(Nx * Ny) 
        print "b = \n"
        print b
        b[0] += - refP
        A[range(1, Nx), range(1, Nx)] += 1
        A[range(-Nx, 0), range(-Nx, 0)] += 1
        A[range(0, Nx*Ny, Nx), range(0, Nx*Ny, Nx)] += 1
        A[range(Nx-1, Nx*Ny, Nx), range(Nx-1, Nx*Ny, Nx)] += 1

    x = u[1:-1, 1:-1].reshape(Nx * Ny)

    err = 1000.
    itr_count = 0
    r = b - A.dot(x)
    p = r
    while (err > tol and itr_count < int(1e8)):

        itr_count += 1

        aph = r.dot(r) / p.dot(A.dot(p))
        aph *= p
        x += aph

        # aph currently is $aph * p = x^{k+1} - x^k$
        err = L2norm(aph)  

        r_new = b - A.dot(x)
        beta = r_new.dot(r_new) / r.dot(r)
        p = r_new + beta * p
        r = r_new
        
        if itr_count % 50 == 0:
            print itr_count, err, aph.max(), aph.min()

    u[1:-1, 1:-1] = x.reshape((Ny, Nx))

    BCHandle[BCtype](u, refP)

    return u, itr_count


