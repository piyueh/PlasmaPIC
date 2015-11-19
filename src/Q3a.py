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

mi = 1.674e-27 # Hydrogen molecular, [Kg]

dt_e = 1e-11 # time step, [s]
dt_i = 1e-5 # time step, [s]


Ne = 10
Ni = 10


Nx = Ny = Nz = 100
Lx = Ly = Lz = 1.


B = 0.1 # [T]
E = 1000.0 # [V/m]

V0_e = numpy.zeros((Ne, 3))
V0_i = numpy.zeros((Ni, 3))

V0_e[:int(Ne/2)] = numpy.random.random_integers(5e4, 5e5, (int(Ne/2), 3)) 
V0_e[int(Ne/2):] = numpy.random.random_integers(-5e5, -5e4, (Ne-int(Ne/2), 3)) 
V0_i[:int(Ni/2)] = numpy.random.random_integers(5e2, 5e3, (int(Ni/2), 3))
V0_i[int(Ni/2):] = numpy.random.random_integers(-5e3, -5e2, (Ni-int(Ni/2), 3))


X0_e = numpy.zeros((Ne, 3)) # initial velocity of electrons, [m/s]
X0_i = numpy.zeros((Ni, 3)) # initial velocity of ions, [m/s]

electrons = numpy.empty(Ne, dtype=Particle)
ions = numpy.empty(Ni, dtype=Particle)


for i in range(Ne):
    electrons[i] = Particle(X0_e[i, :], V0_e[i, :], -e, me, dt_e)
for i in range(Ni):
    ions[i] = Particle(X0_i[i, :], V0_i[i, :], e, mi, dt_e)





field = Field(Nx, Ny, Nz, Lx, Ly, Lz)
field.initE(E)
field.initB(B)

trace_i = numpy.zeros((Ne, 3, 100))
trace_e = numpy.zeros((Ni, 3, 100))

for i in range(100):
    field.updateTho(electrons, ions)

    field.solveE(eps0)

    field.updateParticleProps(electrons)
    field.updateParticleProps(ions)

    for particle in electrons:
        particle.updateVX()

    for particle in ions:
        particle.updateVX()

    for k in range(Ne):
        trace_e[k, :, i] = electrons[k].X.copy()
        if (numpy.any((trace_e[k, :, i]-numpy.array([-Lx/2., -Ly/2., -Lz/2.])) <=0) or 
                numpy.any((trace_e[k, :, i]-numpy.array([Lx/2., Ly/2., Lz/2.])) >= 0)):
            trace_e[k, 0, i+1:] = trace_e[k, 0, i] 
            trace_e[k, 1, i+1:] = trace_e[k, 1, i] 
            trace_e[k, 2, i+1:] = trace_e[k, 2, i] 
            electrons[k].q = 0
            electrons[k].V[:] = 0

    for k in range(Ni):
        trace_i[k, :, i] = ions[k].X.copy()
        if (numpy.any((trace_i[k, :, i]-numpy.array([-Lx/2., -Ly/2., -Lz/2.])) <=0) or 
                numpy.any((trace_i[k, :, i]-numpy.array([Lx/2., Ly/2., Lz/2.])) >= 0)):
            trace_i[k, 0, i+1:] = trace_i[k, 0, i] 
            trace_i[k, 1, i+1:] = trace_i[k, 1, i] 
            trace_i[k, 2, i+1:] = trace_i[k, 2, i] 
            ions[k].q = 0
            ions[k].V[:] = 0
    print(i)





import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

fig = pyplot.figure(figsize=(12,9), dpi=75)
ax = fig.gca(projection='3d')
for i in range(Ne):
    ax.plot(trace_e[i, 0, :], trace_e[i, 1, :], trace_e[i, 2, :], 
            linestyle='-', lw=3, label="Electron"+str(i))
for i in range(Ni):
    ax.plot(trace_i[i, 0, :], trace_i[i, 1, :], trace_i[i, 2, :], 
            linestyle='--', lw=1, label="Ion"+str(i))
ax.set_title('Trace of the Electrons and Ions\nin the Question 3, with dt=1e-11 s', 
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
ax.legend()

pyplot.savefig("../figures/Q3a.png", dpi=75)
pyplot.show()

