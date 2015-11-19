# python 3

##
# @file GridPoint.py
# @brief Definition of GridPoint class
# @author Pi-Yueh Chuang (pychuang@gwu.edu)
# @version alpha
# @date 2015-11-17

import numpy

class GridPoint:
    '''
    Definition of GridPoint class
    '''

    def __init__(self, _x, _y, _z):
        '''
        Input:
            _x: float, x coordinate
            _y: float, y coordinate
            _z: float, z coordinate
        '''

        self.coord = numpy.array([_x, _y, _z])

