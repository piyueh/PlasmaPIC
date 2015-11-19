# python 3

##
# @file Particle.py
# @brief class definition of particle class
# @author Pi-Yueh Chuang (pychuang@gwu.edu)
# @version alpha
# @date 2015-11-16

import numpy

class Particle:
    '''
    Class definition of particle class
    '''

    def __init__(self, _X, _V, _q, _m, _dt):
        '''
        Input: 
            _X: 1x3 numpy array, representing particle position
            _V: 1x3 numpy array, representing particle velocity
            _q: scalar, particle charge
            _m: scalar, particle mass
        '''
        assert type(_X) is numpy.ndarray
        assert _X.size is 3
        assert type(_V) is numpy.ndarray
        assert _V.size is 3
        assert type(_q) is float
        assert type(_m) is float

        self.X = _X.copy()
        self.V = _V.copy()
        self.q = _q
        self.m = _m
        self.dt = _dt
        self.B = numpy.zeros(3)
        self.E = numpy.zeros(3)
        self.C = self.q * self.dt / self.m

    def updateVX(self):
        '''
        coord: instance of Eulerian grid
        '''
        RHS = numpy.array(
                [self.V[0] + self.C * self.E[0] + 0.5 * self.C * self.B[2] * self.V[1],
                 self.V[1] + self.C * self.E[1] - 0.5 * self.C * self.B[2] * self.V[0]])
        A = numpy.array(
                [[1.0, -0.5 * self.C * self.B[2]],
                 [0.5 * self.C * self.B[2], 1.0]])
                 
        self.V[:2] = numpy.linalg.solve(A, RHS)
        self.X = self.X + self.V * self.dt


