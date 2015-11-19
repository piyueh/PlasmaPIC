##
# @file main.py
# @brief solution to the question 2 in homework 5
# @author Pi-Yueh Chuang (pychuang@gwu.edu)
# @version alpha
# @date 2015-11-17

# python 3


import numpy
from Field import Field
from Particle import Particle


e = 1.602176565e-19 # [C]
me = 9.10938291e-31 # [kg]
eps0 = 8.85e-12 # [s^2 C^2 m^-2 Kg^-1]

mi = 6.63e-26#1.674e-27 # Hydrogen molecular, [Kg]

dt_e = 1e-11 # time step, [s]
dt_i = 1e-5 # time step, [s]


Ne = 0
Ni = 1


Nx = Ny = Nz = 100
Lx = Ly = 1.
Lz = 1.05


B = 0.1 # [T]
E = 1000.0 # [V/m]


V0 = numpy.array([1e3, 1e3, 1e3]) # initial velocity, [m/s]
X0 = numpy.array([0., 0., -0.5]) # initial position


electrons = numpy.empty(Ne, dtype=Particle)
ions = numpy.empty(Ni, dtype=Particle)


ions[0] = Particle(X0, V0, e, mi, dt_i)


field = Field(Nx, Ny, Nz, Lx, Ly, Lz)
field.initE(E)
field.initB(B)

trace = numpy.zeros((3, 100))

for i in range(100):
    field.updateTho(electrons, ions)

    field.solveE(eps0)

    field.updateParticleProps(ions)

    for particle in ions:
        particle.updateVX()

    print(i, ions[0].X)
    trace[:, i] = ions[0].X.copy()
    trace[2, i] += 0.5




import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

fig = pyplot.figure(figsize=(12,9), dpi=75)
ax = fig.gca(projection='3d')
ax.plot(trace[0, :], trace[1, :], trace[2, :], 
        lw=2, color='k')
ax.set_title('Trace of the Ion\nin the Question 2', 
        fontsize=18)
#ax.set_xlim(-6e-2, 6e-2)
#ax.set_xticks([-0.05, -0.025, 0, 0.025, 0.05])
#ax.set_xticklabels([-5, -2.5, 0, 2.5, 5])
#ax.xaxis.get_major_formatter().set_offset_string("1e-2")
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xlabel("x (m)", fontsize=16)
#ax.set_ylim(-0.22, -0.1)
#ax.set_yticks([-0.2, -0.175, -0.15, -0.125, -0.1])
#ax.set_yticklabels([-0.2, -0.175, -0.15, -0.125, -0.1])
#ax.yaxis.get_major_formatter().set_offset_string(False)
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_ylabel("y (m)", fontsize=16)
#ax.set_zlim(0, 1.05)
#ax.set_zticks([0, 0.25, 0.5, 0.75, 1])
#ax.set_zticklabels([0, 0.25, 0.5, 0.75, 1])
#ax.zaxis.get_major_formatter().set_offset_string(False)
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_zlabel("z (m)", fontsize=16)

pyplot.savefig("../figures/Q2a.png", dpi=75)
pyplot.show()

