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

    def __init__(self, _X, _V, _q, _m):
        '''
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

    def updateVX(self, grid):
        '''
        coord: instance of Eulerian grid
        '''
        pass


