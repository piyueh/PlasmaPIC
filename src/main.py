##
# @file main.py
# @brief main script
# @author Pi-Yueh Chuang (pychuang@gwu.edu)
# @version alpha
# @date 2015-11-17

# python 3


import numpy
from Field import Field
from Particle import Particle

import matplotlib
from matplotlib import pyplot

e = 1.602176565e-19 # [C]
me = 9.10938291e-31 # [kg]
eps0 = 8.85e-12 # [s^2 C^2 m^-2 Kg^-1]

dt = 1e-11 # time step, [s]

Ne = 1
Ni = 0

Nx = Ny = Nz = 100
Lx = Ly = Lz = 0.2


V0 = numpy.array([1e5, 2e5, 1.5e5]) # initial velocity, [m/s]
X0 = numpy.array([0.1, 0.1, 0.1]) # initial position
B = 0.1 # [T]
E = 1000. # [V/m]

electrons = numpy.empty(Ne, dtype=Particle)
ions = numpy.empty(Ni, dtype=Particle)

electrons[0] = Particle(X0, V0, -e, me, dt)

field = Field(Nx, Ny, Nz, Lx, Ly, Lz)
field.initE(E)
field.initB(B)

trace = numpy.zeros((3, 100))

for i in range(100):
    field.updateTho(electrons, ions)

    #field.solveE(eps0)

    field.updateParticleProps(electrons)

    for particle in electrons:
        particle.updateVX()

    print(electrons[0].X)
    trace[:, i] = electrons[0].X.copy()



#'''
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
ax.plot(trace[0, :], trace[1, :], trace[2, :], label='parametric curve')
ax.legend()

plt.show()
#'''

