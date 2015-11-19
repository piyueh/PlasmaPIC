##
# @file main.py
# @brief this file run the case that only magnetic field exists
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


dt_e = 1e-11 # time step, [s]
dt_i = 1e-5 # time step, [s]


Ne = 1
Ni = 0


Nx = Ny = Nz = 100
Lx = Ly = Lz = 5e-4


B = 0.1 # [T]
E = 0.0 # [V/m]


V0 = numpy.array([1e5, 2e5, 1.5e5]) # initial velocity, [m/s]
X0 = numpy.array([0., 0., 0.]) # initial position


electrons = numpy.empty(Ne, dtype=Particle)
ions = numpy.empty(Ni, dtype=Particle)


electrons[0] = Particle(X0, V0, -e, me, dt_e)


field = Field(Nx, Ny, Nz, Lx, Ly, Lz)
field.initE(E)
field.initB(B)

trace = numpy.zeros((3, 100))

for i in range(100):
    field.updateTho(electrons, ions)

    field.updateParticleProps(electrons)

    for particle in electrons:
        particle.updateVX()

    print(i, electrons[0].X)
    trace[:, i] = electrons[0].X.copy()



import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

fig = pyplot.figure(figsize=(12,9), dpi=75)
ax = fig.gca(projection='3d')
ax.plot(trace[0, :], trace[1, :], trace[2, :], 
        lw=2, color='k')
ax.set_title('One electron under\nmagnetic field independent to plasma', 
        fontsize=18)
ax.set_xlim(-2.5e-5, 0)
ax.set_xticks([-2.5e-5, -2.0e-5, -1.5e-5, -1e-5,-5e-6, 0])
ax.set_xticklabels([-25, -20, -15, -10,-5, 0])
ax.xaxis.get_major_formatter().set_offset_string("1e-6")
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xlabel("x (m)", fontsize=16)
ax.set_ylim(-1e-5, 2e-5)
ax.set_yticks([-10e-6, -5e-6, 0, 5e-6, 10e-6, 15e-6, 20e-6])
ax.set_yticklabels([-10, -5, 0, 5, 10, 15, 20])
ax.yaxis.get_major_formatter().set_offset_string("1e-6")
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_ylabel("y (m)", fontsize=16)
ax.set_zlim(0, 1.5e-4)
ax.set_zticks([0, 5e-5, 10e-5, 15e-5])
ax.set_zticklabels([0, 5, 10, 15])
ax.zaxis.get_major_formatter().set_offset_string("1e-5")
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_zlabel("z (m)", fontsize=16)

ax.set_top_view()

pyplot.savefig("../figures/verify_B=0.1_E=0.png", dpi=75)
pyplot.show()

