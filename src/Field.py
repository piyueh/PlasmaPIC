# python 3

##
# @file Field.py
# @brief class definition of field class
# @author Pi-Yueh Chuang (pychuang@gwu.edu)
# @version alpha
# @date 2015-11-16

import numpy
from numba import jit, void, int32, float64
from GridPoint import GridPoint

class Field:
    '''
    Definition of field class
    '''

    def __init__(self, Nx, Ny, Nz, Lx, Ly, Lz):
        '''
        Input:
            Nx: number of grid points in x direction
            Ny: number of grid points in y direction
            Nz: number of grid points in z direction
            Lx: length in x direction
            Ly: length in y direction
            Lz: length in z direction
        '''
        assert type(Nx) is int
        assert type(Ny) is int
        assert type(Nz) is int
        assert type(Lx) is float
        assert type(Ly) is float
        assert type(Lz) is float

        self.N = numpy.array([Nx, Ny, Nz])
        self.L = numpy.array([Lx, Ly, Lz])
        self.dL = self.L / self.N
        self.GPs = numpy.empty((Nx, Ny, Nz), dtype=GridPoint)
        self.tho = numpy.zeros((Nx, Ny, Nz), dtype=float)
        self.current = numpy.zeros((Nx, Ny, Nz), dtype=float)
        self.E = numpy.zeros((3, Nx, Ny, Nz), dtype=float)
        self.E0 = 0.
        self.B = numpy.zeros((3, Nx, Ny, Nz), dtype=float)
        self.phi = numpy.zeros((Nx, Ny, Nz), dtype=float)

        self._creatGPs()
    

    def _creatGPs(self):
        '''
        Initialize each grid point
        '''
        for i in range(self.N[0]):
            for j in range(self.N[1]):
                for k in range(self.N[2]):
                    self.GPs[i, j, k] = GridPoint(
                            (i + 0.5) * self.dL[0] - 0.5 * self.L[0], 
                            (j + 0.5) * self.dL[1] - 0.5 * self.L[1], 
                            (k + 0.5) * self.dL[2] - 0.5 * self.L[2])


    def initE(self, _E):
        '''
        set constant electric field
        '''
        assert type(_E) is float

        self.E[:, :, :, :] = 0.
        self.E[0, :, :, :] = _E
        self.E0 = _E


    def initB(self, _B):
        '''
        set constant magnetic field
        '''
        assert type(_B) is float

        self.B[:, :, :, :] = 0
        self.B[2, :, :, :] = _B


    def updateTho(self, electrons, ions):
        '''
        '''
        self.tho[:, :, :] = 0.

        for electron in electrons:
            idx = self._calIdx(electron.X)

            for i in range(idx[0]-1, idx[0]+2):
                for j in range(idx[1]-1, idx[1]+2):
                    for k in range(idx[2]-1, idx[2]+2):
            
                        w = self._weighting(electron, numpy.array([i, j, k]))
                        self.tho[i, j, k] += w * electron.q

        for ion in ions:
            idx = self._calIdx(ion.X)

            for i in range(idx[0]-1, idx[0]+2):
                for j in range(idx[1]-1, idx[1]+2):
                    for k in range(idx[2]-1, idx[2]+2):
            
                        w = self._weighting(ion, numpy.array([i, j, k]))
                        self.tho[i, j, k] += w * ion.q


    def _calIdx(self, X):
        '''
        '''
        idx = ((X + self.L / 2.) / self.dL).astype(numpy.int64)
        
        for i in range(3):
            if idx[i] == self.N[i]: 
                idx[i] -= 1

        return idx


    def _weighting(self, particle, idx):
        '''
        '''
        p_min = particle.X - 0.5 * self.dL
        p_Max = particle.X + 0.5 * self.dL

        c_min = self.GPs[idx[0], idx[1], idx[2]].coord - 0.5 * self.dL
        c_Max = self.GPs[idx[0], idx[1], idx[2]].coord + 0.5 * self.dL
    
        d = particle.X - self.GPs[idx[0], idx[1], idx[2]].coord

        if numpy.all(numpy.abs(d) - self.dL <= 0.) :
            AdL = 2 * self.dL - \
                    numpy.maximum(p_Max, c_Max) + \
                    numpy.minimum(p_min, c_min)

            A = numpy.prod(AdL)

            return A / numpy.prod(self.dL)
        else:
            return 0.


    def solveE(self, eps, opt=False):
        '''
        '''
        dx2 = self.dL[0]**2
        dy2 = self.dL[1]**2
        dz2 = self.dL[2]**2
        dd = 2. * (1. / dx2 + 1. / dy2 + 1. / dz2)

        tol = 1e-7
        err = 1000.
        itr_count = 0

        self.phi[:, :, :] = 0.

        while err > tol and itr_count < int(1e1):

            itr_count += 1

            self.phi[1, 1, 1] = 0.

            self.phi[:, 0, :] = self.phi[:, 1, :]
            self.phi[:, -1, :] = self.phi[:, -2, :]
            self.phi[:, :, 0] = self.phi[:, :, 1]
            self.phi[:, :, -1] = self.phi[:, :, -2]

            self.phi[0, :, :] = self.phi[1, :, :] + self.E0 * self.dL[0]
            self.phi[-1, :, :] = self.phi[-2, :, :] - self.E0 * self.dL[0]

            GaussSeidel_jit(self.N[0], self.N[1], self.N[2],
                            dx2, dy2, dz2, dd, eps, self.tho, self.phi)

            err = self._residual(eps)

            if opt:
                print(itr_count, err)

        self.E[0, 1:-1, 1:-1, 1:-1] = - (
            (self.phi[2:, 1:-1, 1:-1] - self.phi[:-2, 1:-1, 1:-1]) / self.dL[0])

        self.E[1, 1:-1, 1:-1, 1:-1] = - (
            (self.phi[1:-1, 2:, 1:-1] - self.phi[1:-1, :-2, 1:-1]) / self.dL[1])

        self.E[2, 1:-1, 1:-1, 1:-1] = - (
            (self.phi[1:-1, 1:-1, 2:] - self.phi[1:-1, 1:-1, :-2]) / self.dL[2])


    def _residual(self, eps):
        '''
        '''
        dx2 = self.dL[0]**2
        dy2 = self.dL[1]**2
        dz2 = self.dL[2]**2

        res = \
                (self.phi[2:, 1:-1, 1:-1] -
                 2. * self.phi[1:-1, 1:-1, 1:-1] +
                 self.phi[:-2, 1:-1, 1:-1]) / dx2 + \
                (self.phi[1:-1, 2:, 1:-1] -
                 2. * self.phi[1:-1, 1:-1, 1:-1] +
                 self.phi[1:-1, :-2, 1:-1]) / dy2 + \
                (self.phi[1:-1, 1:-1, 2:] -
                 2. * self.phi[1:-1, 1:-1, 1:-1] +
                 self.phi[1:-1, 1:-1, :-2]) / dz2 - \
                self.tho[1:-1, 1:-1, 1:-1] / eps
        L2res = numpy.sqrt(numpy.sum(res**2))

        return L2res


    def updateParticleProps(self, particles):
        '''
        '''
        for particle in particles:

            particle.B = numpy.zeros(3)
            particle.E = numpy.zeros(3)
            idx = self._calIdx(particle.X)

            for i in range(idx[0]-1, idx[0]+2):
                for j in range(idx[1]-1, idx[1]+2):
                    for k in range(idx[2]-1, idx[2]+2):
                        w = self._weighting(particle, numpy.array([i, j, k]))
                        particle.E += w * self.E[:, i, j, k]
                        particle.B += w * self.B[:, i, j, k]



@jit(void(int32, int32, int32, float64, float64, float64, float64, float64,
        float64[:, :, :], float64[:, :, :]), nopython=True, nogil=True)
def GaussSeidel_jit(Nx, Ny, Nz, dx2, dy2, dz2, dd, eps, tho, phi):
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            for k in range(1, Nz-1):
                phi[i, j, k] = (
                        (phi[i+1, j, k] + phi[i-1, j, k]) / dx2 + 
                        (phi[i, j+1, k] + phi[i, j-1, k]) / dy2 + 
                        (phi[i, j, k+1] + phi[i, j, k-1]) / dz2 + 
                        tho[i, j, k] / eps) / dd
